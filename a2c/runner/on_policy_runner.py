from tqdm import tqdm
import gymnasium as gym
import numpy as np

class OnPolicyRunner:
    def __init__(self, env, agent, num_episodes):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
    
    def set_agent(self, agent):
        self.agent = agent
    
    def set_env(self, env):
        self.env = env
    
    def run(self):
        # to record episode returns
        return_list = []
        for i in range(10): # 10 pbars by default
            with tqdm(total=int(self.num_episodes/10), desc="Iteration %d"%(i)) as pbar:
                for i_episode in range(int(self.num_episodes/10)): # for each pbar, there are int(num_episodes/10) episodes
                    # each episode
                    episode_return = 0
                    transition_dict = {
                        "states":[],
                        "actions":[],
                        "next_states":[],
                        "rewards":[],
                        "dones":[]
                    }
                    # reset environment
                    state, _ = self.env.reset()
                    terminated, truncated = False, False
                    while not terminated and not truncated: # interaction loop
                        # select action
                        action = self.agent.select_action(state)
                        # step the environment
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        # store transition
                        transition_dict["states"].append(state)
                        transition_dict["actions"].append(action)
                        transition_dict["next_states"].append(next_state)
                        transition_dict["rewards"].append(reward)
                        transition_dict["dones"].append(terminated)
                        # update state
                        state = next_state
                        episode_return += reward
                    # episode finished
                    return_list.append(episode_return)
                    # update agent
                    self.agent.learn(transition_dict)
                    # show mean return of last 10 episodes, every 10 episodes
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({
                            "episode":
                            "%d" % (int(self.num_episodes/10)*i+i_episode+1),
                            "return":
                            "%.3f" % (np.mean(return_list[-10:]))
                        })
                    # update pbar
                    pbar.update(1)
        # return return_list
        return return_list