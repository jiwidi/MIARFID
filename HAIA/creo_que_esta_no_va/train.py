import gym
import math
import random
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from dqn import DuelingDQN, StateHolder, ReplayMemory
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
STATE_SIZE = 4
STATE_W = 84
STATE_H = 84
MEMSIZE = 200000


env = gym.make("BreakoutDeterministic-v4").unwrapped

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print("Is python : {}".format(is_ipython))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(device))

ACTIONS_NUM = env.action_space.n
print("Number of actions : {}".format(ACTIONS_NUM))

resize = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((STATE_W, STATE_H), interpolation=Image.CUBIC),
        T.ToTensor(),
    ]
)


def get_screen():
    screen = env.render(mode="rgb_array")
    screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
    screen = screen[30:195, :]
    screen = np.ascontiguousarray(screen, dtype=np.uint8).reshape(
        screen.shape[0], screen.shape[1], 1
    )
    return (
        resize(screen).mul(255).type(torch.ByteTensor).to(device).detach().unsqueeze(0)
    )


policy_net = DuelingDQN().to(device)
target_net = DuelingDQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def select_action(state, eps_threshold):
    global steps_done
    sample = random.random()
    if sample > eps_threshold and state is not None:
        with torch.no_grad():
            return policy_net(state.float()).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(ACTIONS_NUM)]], device=device, dtype=torch.long
        )


BATCH_SIZE = 32
GAMMA = 0.99


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.uint8,
    )

    non_final_next_states = (
        torch.cat([s for s in batch.next_state if s is not None]).float().to(device)
    )
    state_batch = torch.cat(batch.state).float().to(device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_action = policy_net(non_final_next_states).detach().max(1)[1].view(-1, 1)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).detach().gather(1, next_state_action).view(-1)
    )

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    del non_final_mask
    del next_state_action
    del non_final_next_states
    del state_batch
    del action_batch
    del reward_batch
    del state_action_values
    del next_state_values
    del expected_state_action_values
    del loss


mean_size = 100
mean_step = 1
train_rewards = []


NUM_EPISODES = 2000

OPTIMIZE_MODEL_STEP = 4
TARGET_UPDATE = 10000

STEPS_BEFORE_TRAIN = 50000

EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000
eps_threshold = EPS_START

policy_net.train()
target_net.eval()

state_holder = StateHolder()
memory = ReplayMemory()
test_rewards = []
train_rewards = []

steps_done = 0
optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)


NUM_EPISODES = 50000
pbar = tqdm(range(NUM_EPISODES))

for e in tqdm(range(NUM_EPISODES)):
    env.reset()
    lives = 5
    ep_rewards = []
    state_holder.push(get_screen())

    for t in count():
        state = state_holder.get()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )
        action = select_action(state, eps_threshold)
        steps_done += 1

        _, reward, done, info = env.step(action.item())
        life = info["ale.lives"]
        ep_rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        state_holder.push(get_screen())
        next_state = state_holder.get()

        if not done:
            new_reward = reward
            next_state, lives = (None, life) if life < lives else (next_state, lives)
            memory.push(state.to("cpu"), action, next_state, new_reward)
            state = next_state
        else:
            next_state = None
            new_reward = torch.zeros_like(reward)
            memory.push(state.to("cpu"), action, next_state, new_reward)
            state = next_state

        if (steps_done > STEPS_BEFORE_TRAIN) and steps_done % OPTIMIZE_MODEL_STEP == 0:
            BATCH_SIZE = 32
            optimize_model()
        if t > 18000:
            break

        if steps_done % TARGET_UPDATE == 0:
            # print("Target net updated!")
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            train_rewards.append(np.sum(ep_rewards))
            break
        pbar.set_description(
            f"Episode: {e} Mean score: {np.mean(train_rewards[-100:]):.2f} 10 ep.mean score: {np.mean(train_rewards[-10:]):.2f}"
        )
