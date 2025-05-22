import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports all our hyperparameters from the other file
from hyperparams import Hyperparameters as params

# stable_baselines3 have wrappers that simplifies 
# the preprocessing a lot, read more about them here:
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, #Setes the reward to -1, 0 or 1
    EpisodicLifeEnv, #Ends game if life lost even if game itself isnt over
    FireResetEnv, #Fires at game start so it doesnt get stuck
    MaxAndSkipEnv, #Skips frames to make training faster
    NoopResetEnv, #Resets the environment to a random state by adding noops at the start
)
from stable_baselines3.common.buffers import ReplayBuffer
from gym.wrappers import RecordEpisodeStatistics, RecordVideo

# Creates our gym environment and with all our wrappers.
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if capture_video and idx == 0 and video_trigger is not None:
            video_folder = os.path.join("videos", run_name)
            os.makedirs(video_folder, exist_ok=True)
            env = RecordVideo(env, video_folder=video_folder, episode_trigger=video_trigger)
        
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Look at Section 4.1 in the paper for help: https://arxiv.org/pdf/1312.5602v1.pdf
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.network = nn.Sequential(
            #Extract features with CNN
            nn.Conv2d(4, 16, 8, 4), #input, output, kernel size, stride
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), #input, output, kernel size, stride
            nn.ReLU(),
            #Compress the features to a vector
            nn.Flatten(),
            nn.Linear(9*9*32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            #Output the Q values for each action
            nn.Linear(512, env.single_action_space.n)                
        )

    #Takes in the state and outputs the Q values for each action via network
    def forward(self, x):
        return self.network(x / 255.0)

#Determines epsilon value that is used to determine if we should do a random action or not
#Used to counter Exploration vs Explotation problem
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic #Ensures repeatable results (force gpu to run deterministically)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def video_trigger(episode_id): # Make sure we record every 250 episode (episode = full game)
        return episode_id % 250 == 0

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(params.env_id, params.seed, 0, params.capture_video, run_name)]) #Creates vector of envs (We only use 1 env but could have used 4 to speed up the process)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported" #Makes sure we have discrete actions since DQN needs fixed list of actions

    q_network = QNetwork(envs).to(device) #Creates our Q network and target network
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict()) #Copy the weights of the Q network to the target network
    
    # We’ll be using experience replay memory for training our DQN. 
    # It stores the transitions that the agent observes, allowing us to reuse this data later. 
    # By sampling from it randomly, the transitions that build up a batch are decorrelated. 
    # It has been shown that this greatly stabilizes and improves the DQN training procedure.
    rb = ReplayBuffer(
        params.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs = envs.reset() #Start first episode
    for global_step in range(params.total_timesteps):
        # Here we get epsilon for our epislon greedy.
        epsilon = linear_schedule(params.start_e, params.end_e, params.exploration_fraction * params.total_timesteps, global_step)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()]) # ADDED
        else:
            q_values = q_network(torch.tensor(obs, dtype=torch.float32).to(device)) # ADDED
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Take a step in the environment
        next_obs, rewards, dones, infos = envs.step(actions)

        # Here we print our reward.
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                break

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        # Here we store the transitions in replay buffer
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        obs = next_obs
        # Training 
        if global_step > params.learning_starts:
            if global_step % params.train_frequency == 0:
                # Sample random minibatch of transitions from D
                data = rb.sample(params.batch_size)
                # You can get data with:
                # data.observation, data.rewards, data.dones, data.actions

                with torch.no_grad():
                    # Now we calculate the y_j for non-terminal phi.
                    target_q_values = target_network(data.next_observations) #Gets vector of Q values for every next move from target network
                    target_max, _ = target_q_values.max(dim=1) #Get single best Q value (future reward)
                    td_target = data.rewards.flatten() + params.gamma * target_max * (1 - data.dones.flatten()) #Calculate the target value (y_j) with the Bellman equation
                    #Reward + gamma * max Q of next state with a check if its terminal state or not

                q_values = q_network(data.observations) #Batch of predicted Q values from the Q network
                old_val = q_values.gather(1, data.actions).squeeze() #Pick the Q value of the action we took
                loss = F.mse_loss(old_val, td_target) #Check difference between the Q value we took and the target value (y_j) to train on

                # perform our gradient decent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % params.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        params.tau * q_network_param.data + (1.0 - params.tau) * target_network_param.data
                    )

    if params.save_model: #Saves weights of the model
        model_path = f"runs/{run_name}/{params.exp_name}_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
