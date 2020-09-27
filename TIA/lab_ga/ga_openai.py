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
import random
from tqdm import tqdm
import pdb


num_cores = multiprocessing.cpu_count()
enviorment = "CartPole-v1"
game_actions = 2  # 2 actions possible: left or right
torch.set_grad_enabled(False)  # disable gradients as we will not use them
num_agents = 500  # initialize N number of agents
top_limit = 200
generations = 1000000


class CartPoleAI(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(4, 128, bias=True), nn.ReLU(), nn.Linear(128, 2, bias=True), nn.Softmax(dim=1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 128, bias=True),
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


def join_parents(parent1):
    import pdb

    pdb.set_trace()


def selection_ruleta(agents, fitness_list):
    normalized_fitness = [float(i) / sum(fitness_list) for i in fitness_list]
    return random.choices(population=agents, weights=normalized_fitness, k=len(agents))


def selection_top(agents, fitness_list):
    sorted_parent_indexes = np.argsort(fitness_list)[::-1][:top_limit]
    top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
    return random.choices(population=top_agents, k=len(agents))


def join_cross(parents):
    children = []
    for parent1, parent2 in zip(parents[0::2], parents[1::2]):
        copy_parent1 = copy.deepcopy(parent1)
        copy_parent2 = copy.deepcopy(parent2)
        total = len(list(copy_parent1.parameters()))
        i = 0
        for param1, param2 in zip(copy_parent1.parameters(), copy_parent2.parameters()):
            if i < total / 2:
                param1.data = param1.data * 1
                param2.data = param2.data * 1
            else:
                param1.data = param2.data * 1
                param2.data = param1.data * 1
            i += 1
        children.append(copy_parent1)
        children.append(copy_parent2)
    return children


def not_join(parents):
    return parents


def return_children(
    agents,
    fitness_list,
    elite_index,
    selection_function,
    join_function,
):
    children_agents = []
    # Select parents
    selected_parents = selection_function(agents, fitness_list)
    # Cuzamos los padres dado el metodo que hayamos elegido
    children_agents = join_function(selected_parents)
    # Mutamos los hijos
    children_agents = Parallel(n_jobs=num_cores)(delayed(mutate)(i) for i in children_agents)

    return children_agents  # , elite_index


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
        fitness = run_agents_n_times(agents, 3)
        sorted_parent_indexes = np.argsort(fitness)[::-1][:top_limit]
        top_fitness = [fitness[best_parent] for best_parent in sorted_parent_indexes]

        print(
            "Generation {0:.3g} | Mean fitness: {1:.3g} | Mean reward of top 5: {2:.4g}".format(
                generation,
                np.mean(fitness),
                np.mean(top_fitness[:5]),
            )
        )

        # setup an empty list for containing children agents
        children_agents = return_children(agents, fitness, elite_index, selection_ruleta, join_cross)

        # kill all agents, and replace them with their children
        agents = children_agents

    play_agent(agents[elite_index])


if __name__ == "__main__":
    main()