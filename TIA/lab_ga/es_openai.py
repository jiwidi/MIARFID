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
import pandas as pd
import math
import copy
import random
from tqdm import tqdm


num_cores = multiprocessing.cpu_count()
enviorment = "CartPole-v1"
game_actions = 2  # 2 actions possible: left or right
torch.set_grad_enabled(False)  # disable gradients as we will not use them
np.random.seed(42)
MAX_ITER = 1000


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


def fitness(agent):
    agent.eval()
    r = 0
    s = 0
    rs = []
    for _ in range(3):
        env = gym.make(enviorment)
        observation = env.reset()
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
        rs.append(r)
    return sum(rs) / len(rs)  # Medium score of 3 runs


def neighbor(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        param.data += mutation_power * torch.randn_like(param)
    return child_agent


def run_es_experiment(T):
    sol = return_random_agents(1)[0]
    new_agent = []
    T = 100000000000000

    for i in range(20000):
        f_s = fitness(sol)
        new_agent = copy.copy(sol)
        new_agent = neighbor(new_agent)

        f_ns = fitness(new_agent)
        dif = f_s - f_ns
        if dif < 0:  # si mejora el fitness
            sol = new_agent
            print("-------------------")
            print("NEW BEST SOLUTION: (ITER)", i)
            # print(f_ns)
            print(f_s)

        else:
            if T > 0 and math.exp(-dif / T) > 0.1:
                sol = new_agent

                print("-------------------")
                print("NEW NOT BEST SOLUTION: (ITER)", i)
                print(f_ns)
            T = T / (i + 0.01 * T)


def actualizarTemperatura(iteraciones, k, temperatura):
    return temperatura / (1 + k * temperatura)


def enfriamiento(solucionInicial, temperaturaInicial, k):
    iteraciones = 0
    solucionActual = copy.copy(solucionInicial)
    mejorSolucion = copy.copy(solucionActual)
    temperatura = temperaturaInicial
    pbar = tqdm()
    while iteraciones < 10000:
        solucionNueva = copy.copy(solucionActual)
        solucionNueva = neighbor(solucionNueva)

        fitness_actual = fitness(solucionActual)
        fitness_nueva = fitness(solucionNueva)
        incrementoFitness = -(fitness_actual - fitness_nueva)
        if incrementoFitness > 0:
            solucionActual = copy.copy(solucionNueva)

            if fitness_nueva > fitness_mejor:
                mejorSolucion = copy.copy(solucionActual)
                fitness_mejor = fitness(mejorSolucion)
                # print("mejor solucion encontrada en ", fitness(mejorSolucion))

        else:
            if random.random() < math.exp(incrementoFitness / temperatura):
                solucionActual = copy.copy(solucionNueva)

        iteraciones += 1
        temperatura = actualizarTemperatura(iteraciones, k, temperatura)
        # TQDM
        pbar.update(1)
        pbar.set_description(
            f"Enfriamento frio best_fitness = {fitness_mejor:.2f} Parametros T = {temperaturaInicial}, K = {k}"
        )
        if fitness_mejor > 240:
            pbar.close()
            break
    return fitness_mejor, iteraciones


def experimentos():
    solucionInicial = return_random_agents(1)[0]
    datos = {"temperatura": [], "k": [], "iteraciones": [], "fitnessmedia": []}
    for temperature in [1, 10, 100, 1000, 10000]:
        for k in [0.001, 0.01, 0.1, 1, 10]:
            r = Parallel(n_jobs=num_cores)(
                delayed(enfriamiento)(solucionInicial, temperature, k) for i in range(20)  # tqdm(agents, leave=False)
            )
            sol_fitness = [i[0] for i in r]
            iteraciones = [i[1] for i in r]
            sol_fitness = sum(sol_fitness) / len(sol_fitness)
            iteraciones = sum(iteraciones) / len(iteraciones)

            datos["temperatura"].append(temperature)
            datos["k"].append(k)
            datos["fitnessmedia"].append(sol_fitness)
            datos["iteraciones"].append(iteraciones)
    resultadosDataFrame = pd.DataFrame(datos)
    resultadosDataFrame.to_pickle("exportEvaluacionEnfriamiento.pkl")


def main():
    # run_es_experiment(1)
    experimentos()


if __name__ == "__main__":
    main()