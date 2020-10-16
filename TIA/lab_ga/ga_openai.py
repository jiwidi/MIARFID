#!/usr/bin/env python
# coding: utf-8
import pickle
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
num_agents = 200  # initialize N number of agents
top_limit = 20
generations = 10000


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
            action = np.random.choice(
                range(game_actions), 1, p=output_probabilities
            ).item()
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
        delayed(return_average_score)(i, runs)
        for i in agents  # tqdm(agents, leave=False)
    )
    return agents_avg_scores


def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = (
        0.02  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    )
    for param in child_agent.parameters():
        param.data += mutation_power * torch.randn_like(param)
    return child_agent


def selection_ruleta(agents, fitness_list, k=0.6 * num_agents):
    if k < 0:
        k = len(agents)  # return a full population
    normalized_fitness = [float(i) / sum(fitness_list) for i in fitness_list]
    selection = random.choices(population=agents, weights=normalized_fitness, k=k)

    return selection


def selection_top(agents, fitness_list, k=0.2 * num_agents):
    if k < 0:
        k = len(agents)  # return a full population
    sorted_parent_indexes = np.argsort(fitness_list)[::-1][:k]
    top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
    selection = top_agents
    return selection


def join_cross_new(parents, k=-1):
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


def join_cross_old(parents):
    children = []
    for parent1, parent2 in zip(parents[0::2], parents[1::2]):
        copy_parent1 = copy.deepcopy(parent1)
        copy_parent2 = copy.deepcopy(parent2)
        total = len(list(copy_parent1.parameters()))
        i = 0
        for param1, param2 in zip(copy_parent1.parameters(), copy_parent2.parameters()):
            if i < total / 2:
                param1.data = param1.data * 1
            else:
                param1.data = param2.data * 1
            i += 1
        children.append(copy_parent1)
        children.append(copy_parent2)
    return children


def not_join(parents):
    children = parents
    return children


def replace_generational(agents, fitness_list, children_agents):
    return random.choices(children_agents, k=len(agents))


def replace_state_stationary(agents, fitness_list, children_agents):
    sorted_parents_indexes = np.argsort(fitness_list)[::-1][:top_limit]
    sorted_parents = [agents[best_parent] for best_parent in sorted_parents_indexes]
    selection = sorted_parents[: len(agents) - len(agents)] + children_agents
    return selection


def doom_day(agents, fitness_list, children_agents):
    fitness = run_agents_n_times(children_agents, 3)
    sorted_agents_indexes = np.argsort(fitness_list)[::-1]
    selection = [children_agents[sorted_agents_indexes[0]]]
    selection = selection + return_random_agents(len(agents) - 1)
    return selection


def select_agents(agents, fitness_list, mode="top", k=1):
    if mode == "top":
        return selection_top(agents, fitness_list, k)
    elif mode == "ruleta":
        return selection_ruleta(agents, fitness_list, k)
    else:
        assert 1 == 0, "Mode not supported"


def join(parents, mode="cross"):
    if mode == "cross":
        return join_cross_new(parents)
    elif mode == "none":
        return not_join(parents)
    elif mode == "cross-old":
        return join_cross_old(parents)
    else:
        assert 1 == 0, "Mode not supported"


def replace(agents, fitness_list, children_agents, replace_mode):
    if replace_mode == "generational":
        return replace_generational(agents, fitness_list, children_agents)
    elif replace_mode == "state":
        return replace_state_stationary(agents, fitness_list, children_agents)
    elif replace_mode == "doom-day":
        return doom_day(agents, fitness_list, children_agents)
    return children_agents


def return_children(
    agents,
    fitness_list,
    selection_mode,
    join_mode,
    replace_mode,
    num_agent_randoms=0,
    k=-1,
):
    children_agents = []
    # Select parents
    selected_parents = select_agents(agents, fitness_list, selection_mode, k)
    # Cuzamos los padres dado el metodo que hayamos elegido
    children_agents = join(selected_parents, join_mode)
    # Add extra random agents
    # Mutamos los hijos
    children_agents = Parallel(n_jobs=num_cores)(
        delayed(mutate)(i) for i in children_agents
    )
    # Reemplazo
    children_agents = replace(agents, fitness_list, children_agents, replace_mode)
    assert len(agents) == len(
        children_agents
    ), "Error in genetic loop, missmatch in populations length"
    return children_agents


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
            action = np.random.choice(
                range(game_actions), 1, p=output_probabilities
            ).item()
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
    selection_modes = ["ruleta", "top"]
    join_modes = ["cross", "none", "cross-old"]
    replace_modes = ["generational", "state", "doom-day"]
    num_total_agents = [20, 50, 100, 200, 500, 1000]
    for num_total_agent in num_total_agents:
        results = {}
        num_agents = num_total_agent
        agents = return_random_agents(num_agents)
        for replace_mode in replace_modes:
            for selection_mode in selection_modes:
                for join_mode in join_modes:
                    agents = return_random_agents(num_agents)
                    mean_fitness_history = []
                    pbar = tqdm(range(generations), leave=False)
                    for generation in pbar:
                        fitness = run_agents_n_times(agents, 3)
                        sorted_parent_indexes = np.argsort(fitness)[::-1][:top_limit]
                        top_fitness = [
                            fitness[best_parent]
                            for best_parent in sorted_parent_indexes
                        ]
                        mean_fitness_history.append(np.mean(top_fitness[:5]))
                        if (np.mean(top_fitness[:5])) > 249:
                            print(
                                f"Selec-{selection_mode} join-{join_mode} replace-{replace_mode} total_agents {num_total_agent} converged at generation {generation}"
                            )
                            break
                        elif generation == 100 and np.mean(fitness) < 25:
                            print("Not converging, aborting")
                            print()
                            break
                        pbar.set_description(
                            f"Selec-{selection_mode} join {join_mode} replace-{replace_mode} total_agents {num_total_agent} | Gen {generation} - Top5fitness:{np.mean(top_fitness[:5]):.2f} - Mean fitness:{np.mean(fitness):.2f}"
                        )
                        # setup an empty list for containing children agents
                        children_agents = return_children(
                            agents=agents,
                            fitness_list=fitness,
                            selection_mode=selection_mode,
                            join_mode=join_mode,
                            replace_mode=replace_mode,
                            num_agent_randoms=0,
                        )

                        # kill all agents, and replace them with their children
                        agents = children_agents
                    results[(selection_mode, join_mode)] = mean_fitness_history
        with open(
            f"results/results_ga_agentsperpop{num_agents}.pickle", "wb"
        ) as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()