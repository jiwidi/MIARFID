import random
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import gym

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, output_dim)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)


final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.Tensor(state)
            steps_done += 1
            q_calc = model(state)
            node_activated = int(torch.argmax(q_calc))
            return node_activated
    else:
        node_activated = random.randint(0,1)
        steps_done += 1
        return node_activated


class ReplayMemory(object): # Stores [state, reward, action, next_state, done]

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [[],[],[],[],[]]

    def push(self, data):
        """Saves a transition."""
        for idx, point in enumerate(data):
            #print("Col {} appended {}".format(idx, point))
            self.memory[idx].append(point)

    def sample(self, batch_size):
        rows = random.sample(range(0, len(self.memory[0])), batch_size)
        experiences = [[],[],[],[],[]]
        for row in rows:
            for col in range(5):
                experiences[col].append(self.memory[col][row])
        return experiences

    def __len__(self):
        return len(self.memory[0])


input_dim, output_dim = 4, 2
model = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(model.state_dict())
target_net.eval()
tau = 100
discount = 0.99

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

memory = ReplayMemory(65536)
BATCH_SIZE = 128


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    experiences = memory.sample(BATCH_SIZE)
    state_batch = torch.Tensor(experiences[0])
    action_batch = torch.LongTensor(experiences[1]).unsqueeze(1)
    reward_batch = torch.Tensor(experiences[2])
    next_state_batch = torch.Tensor(experiences[3])
    done_batch = experiences[4]

    pred_q = model(state_batch).gather(1, action_batch)

    next_state_q_vals = torch.zeros(BATCH_SIZE)

    for idx, next_state in enumerate(next_state_batch):
        if done_batch[idx] == True:
            next_state_q_vals[idx] = -1
        else:
            # .max in pytorch returns (values, idx), we only want vals
            next_state_q_vals[idx] = (target_net(next_state_batch[idx]).max(0)[0]).detach()


    better_pred = (reward_batch + next_state_q_vals).unsqueeze(1)

    loss = F.smooth_l1_loss(pred_q, better_pred)
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


points = []
losspoints = []


env = gym.make('CartPole-v0')
for i_episode in range(5000):
    observation = env.reset()
    episode_loss = 0
    if i_episode % tau == 0:
        target_net.load_state_dict(model.state_dict())
    for t in range(1000):
        #env.render()
        state = observation
        action = select_action(observation)
        observation, reward, done, _ = env.step(action)

        if done:
            next_state = [0,0,0,0]
        else:
            next_state = observation

        memory.push([state, action, reward, next_state, done])
        episode_loss = episode_loss + float(optimize_model())
        if done:
            if(t==200):
                print("Problem solved at Episode {} finished after {} timesteps".format(i_episode, t+1))
                break

            points.append((i_episode, t+1))
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Avg Loss: ", episode_loss / (t+1))
            losspoints.append((i_episode, episode_loss / (t+1)))
            if (i_episode % 100 == 0):
                eps = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
                print(eps)
            if ((i_episode+1) % 5001 == 0):
                save = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(save, "models/DQN_target_" + str(i_episode // 5000) + ".pth")
            break
env.close()




x = [coord[0] * 100 for coord in points]
y = [coord[1] for coord in points]

plt.plot(x, y)
plt.xlabel('Steps', fontsize=18)
plt.ylabel('Score', fontsize=16)
plt.savefig('score_cartpole.png')