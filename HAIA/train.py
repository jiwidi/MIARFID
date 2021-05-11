from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from utils import make_atari, wrap_deepmind
from tqdm import tqdm

from model import DQN, ReplayMemory, fp, ActionSelector

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # if gpu is to be used


def optimize_model(train, optimizer, memory, config, policy_net, target_net):
    if not train:
        return
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(
        config["BATCH_SIZE"]
    )

    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (nq * config["GAMMA"]) * (
        1.0 - done_batch[:, 0]
    ) + reward_batch[:, 0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def evaluate(
    step,
    policy_net,
    device,
    env,
    n_actions,
    config,
    train,
    eps=0.05,
    num_episode=5,
):
    env = wrap_deepmind(env)
    sa = ActionSelector(eps, eps, policy_net, config["EPS_DECAY"], n_actions, device)
    e_rewards = []
    q = deque(maxlen=5)
    for i in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(10):  # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, train)
            n_frame, reward, done, info = env.step(action)
            n_frame = fp(n_frame)
            q.append(n_frame)

            e_reward += reward
        e_rewards.append(e_reward)

    f = open("file.txt", "a")
    f.write(
        "%f, %d, %d\n" % (float(sum(e_rewards)) / float(num_episode), step, num_episode)
    )
    f.close()
    return float(sum(e_rewards)) / float(num_episode)


def main():

    env_name = "Breakout"
    env_raw = make_atari("{}NoFrameskip-v4".format(env_name))
    env = wrap_deepmind(
        env_raw, frame_stack=False, episode_life=True, clip_rewards=True
    )

    c, h, w = fp(env.reset()).shape
    n_actions = env.action_space.n
    print(f"Env {env_name} with {n_actions} actions")

    # 4. Network reset
    policy_net = DQN(n_actions, device).to(device)
    target_net = DQN(n_actions, device).to(device)
    policy_net.apply(policy_net.init_weights)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    with open("config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0000625, eps=1.5e-4)

    # replay memory and action selector
    memory = ReplayMemory(config["M_SIZE"], [5, h, w], n_actions, device)
    sa = ActionSelector(
        config["EPS_START"],
        config["EPS_END"],
        policy_net,
        config["EPS_DECAY"],
        n_actions,
        device,
    )

    q = deque(maxlen=5)
    done = True
    episode = 0
    episode_len = 0

    progressive = tqdm(
        range(config["NUM_STEPS"]),
        total=config["NUM_STEPS"],
        leave=False,
        unit="b",
    )
    rewards = []
    for step in progressive:
        if done:  # life reset !!!
            episode += 1
            env.reset()
            sum_reward = 0
            episode_len = 0
            img, _, _, _ = env.step(1)  # BREAKOUT specific !!!
            for i in range(10):  # no-op
                n_frame, _, _, _ = env.step(0)
                n_frame = fp(n_frame)
                q.append(n_frame)

        train = len(memory) > 50000
        # Select and perform an action
        state = torch.cat(list(q))[1:].unsqueeze(0)
        action, eps = sa.select_action(state, train)
        n_frame, reward, done, info = env.step(action)
        rewards.append(reward)
        n_frame = fp(n_frame)

        # 5 frame as memory
        q.append(n_frame)
        memory.push(
            torch.cat(list(q)).unsqueeze(0), action, reward, done
        )  # here the n_frame means next frame from the previous time step
        episode_len += 1

        # Perform one step of the optimization (on the target network)
        if step % config["POLICY_UPDATE"] == 0:
            optimize_model(train, optimizer, memory, config, policy_net, target_net)

        # Update the target network, copying all weights and biases in DQN
        if step % config["TARGET_UPDATE"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % config["EVALUATE_FREQ"] == 0:
            last_reward = evaluate(
                step,
                policy_net,
                device,
                env_raw,
                n_actions,
                config=config,
                eps=0.05,
                num_episode=episode,
                train=train,
            )

        progressive.set_description(
            f"Step {step}, Mean training rewards {np.mean(rewards[-10:]):.2f}, Test reward {last_reward}"
        )


if __name__ == "__main__":
    main()
