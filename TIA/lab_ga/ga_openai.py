#!/usr/bin/env python
# coding: utf-8
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
num_agents = 500  # initialize N number of agents
top_limit = 200
generations = 1000


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 6 * 6)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


class CartPoleAI(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(4, 128, bias=True), nn.ReLU(), nn.Linear(128, 2, bias=True), nn.Softmax(dim=1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, num_actions, bias=True),
            nn.Softmax(dim=1)
            # )
        )

    def forward(self, inputs):
        x = self.fc(inputs)
        return x


def init_weights(m):
    if (isinstance(m, nn.Linear)) | (isinstance(m, nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agent = CartPoleAI(4, 2)
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
    agents_avg_scores = Parallel(n_jobs=num_cores)(
        delayed(return_average_score)(i, runs) for i in tqdm(agents, leave=False)
    )
    # agents_avg_scores = []
    # for agent in tqdm(agents):
    #     agents_avg_scores = agents_avg_scores + [return_average_score(agent, runs)]
    return agents_avg_scores


def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        param.data += mutation_power * torch.randn_like(param)
    return child_agent


def return_children(agents, sorted_parent_indexes, elite_index):
    children_agents = []
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
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent


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


def main():
    agents = return_random_agents(num_agents)
    elite_index = None
    for generation in range(generations):
        rewards = run_agents_n_times(agents, 3)
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit]
        top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]

        print(
            "Generation {0:.3g} | Mean rewards: {1:.3g} | Mean reward of top 5: {2:.4g}".format(
                generation,
                np.mean(rewards),
                np.mean(top_rewards[:5]),
            )
        )

        # setup an empty list for containing children agents
        children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)

        # kill all agents, and replace them with their children
        agents = children_agents

    play_agent(agents[elite_index])


if __name__ == "__main__":
    main()