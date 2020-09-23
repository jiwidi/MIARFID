#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from gym.wrappers import Monitor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
from tqdm import tqdm

num_cores = multiprocessing.cpu_count()
enviorment = "CartPole-v1"
game_actions = 2  # 2 actions possible: left or right
torch.set_grad_enabled(False)  # disable gradients as we will not use them
num_agents = 50000  # initialize N number of agents
top_limit = 200
generations = 1000


class CartPoleAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128, bias=True), nn.ReLU(), nn.Linear(128, 2, bias=True), nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        x = self.fc(inputs)
        return x


def init_weights(m):
    # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
    # nn.Conv2d bias is of shape [16] i.e. # number of filters
    # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
    # nn.Linear bias is of shape [32] i.e. # number of output features
    if (isinstance(m, nn.Linear)) | (isinstance(m, nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agent = CartPoleAI()
        for param in agent.parameters():
            param.requires_grad = False
        init_weights(agent)
        agents.append(agent)

    return agents


def run_agents(agents):
    reward_agents = []
    env = gym.make(enviorment)
    for agent in agents:
        agent.eval()
        observation = env.reset()
        r = 0
        s = 0
        for _ in range(250):
            inp = torch.tensor(observation).type("torch.FloatTensor").view(1, -1)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
            new_observation, reward, done, info = env.step(action)
            r = r + reward
            s = s + 1
            observation = new_observation
            if done:
                break
        reward_agents.append(r)

    return reward_agents


def return_average_score(agent, runs):
    score = 0.0
    for i in range(runs):
        score += run_agents([agent])[0]
    return score / runs


def run_agents_n_times(agents, runs):
    agents_avg_scores = Parallel(n_jobs=num_cores)(delayed(return_average_score)(i, runs) for i in agents)
    return agents_avg_scores


def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        if len(param.shape) == 4:  # weights of Conv2D
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    for i2 in range(param.shape[2]):
                        for i3 in range(param.shape[3]):
                            param[i0][i1][i2][i3] += mutation_power * np.random.randn()
        elif len(param.shape) == 2:  # weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1] += mutation_power * np.random.randn()
        elif len(param.shape) == 1:  # biases of linear layer or conv layer
            for i0 in range(param.shape[0]):
                param[i0] += mutation_power * np.random.randn()
    return child_agent


def return_children(agents, sorted_parent_indexes, elite_index):

    children_agents = []

    # first take selected parents from sorted_parent_indexes and generate N-1
    # children
    # for i in range(len(agents) - 1):
    #     selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
    #     children_agents.append(mutate(agents[selected_agent_index]))
    # agents = tqdm(agents)
    children_agents = Parallel(n_jobs=num_cores)(
        delayed(mutate)(agents[sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]]) for i in agents
    )
    # now add one elite
    elite_child = add_elite(agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index = len(children_agents) - 1  # it is the last one

    return children_agents, elite_index


def add_elite(agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):

    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
    if elite_index is not None:
        candidate_elite_index = np.append(candidate_elite_index, [elite_index])

    top_score = None
    top_elite_index = None

    for i in candidate_elite_index:
        score = return_average_score(agents[i], runs=5)
        if top_score is None:
            top_score = score
            top_elite_index = i
        elif score > top_score:
            top_score = score
            top_elite_index = i

    print("Elite selected with index ", top_elite_index, " and score", top_score)
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def play_agent(agent):
    try:  # try and exception block because, render hangs if an erorr occurs, we must do env.close to continue working
        env = gym.make(enviorment)
        env_record = Monitor(env, "./video", force=True)
        observation = env_record.reset()
        last_observation = observation
        r = 0
        for _ in range(250):
            env_record.render()
            inp = torch.tensor(observation).type("torch.FloatTensor").view(1, -1)
            print(inp.shape)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
            new_observation, reward, done, info = env_record.step(action)
            r = r + reward
            observation = new_observation
            if done:
                break

        env_record.close()
        print("Rewards: ", r)

    except Exception as e:
        env_record.close()
        print(e.__doc__)
        print(e.message)


def main():
    agents = return_random_agents(num_agents)  # How many top agents to consider as parents
    elite_index = None
    for generation in range(generations):
        rewards = run_agents_n_times(agents, 3)
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit]
        top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]

        print(
            "Generation ",
            generation,
            " | Mean rewards: ",
            np.mean(rewards),
            " | Mean of top 5: ",
            np.mean(top_rewards[:5]),
        )

        # setup an empty list for containing children agents
        children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)

        # kill all agents, and replace them with their children
        agents = children_agents

    play_agent(agents[elite_index])


if __name__ == "__main__":
    main()