import random
import gym
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
import imageio
import os
from datetime import datetime

reward_sums = []
reward_running_avgs = []
all_rewards = []
gif_frames = []

os.makedirs("logs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.gamma = 0.99
        self.eps_clip = 0.1
        self.layers = nn.Sequential(
            nn.Linear(6000, 512), nn.ReLU(),
            nn.Linear(512, 2),
        )

    def state_to_tensor(self, I):
        if I is None:
            return torch.zeros(1, 6000).to(device)
        if isinstance(I, tuple):
            I = I[0]
        I = I[35:185]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0).to(device)

    def pre_process(self, x, prev_x):
        return self.state_to_tensor(x) - self.state_to_tensor(prev_x)

    def convert_action(self, action):
        return action + 2

    def forward(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cpu().numpy())
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob

        vs = np.array([[1., 0.], [0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()]).to(device)
        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        return torch.mean(loss)

@torch.no_grad()
def perturb_params(params, grad_eps=1e-12, rho=0.05):
    grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                        for p in params if p.grad is not None]))
    scale = rho / (grad_norm + grad_eps)
    for p in params:
        if p.grad is not None:
            p._backup = p.data.clone()
            p.add_(p.grad, alpha=scale)

@torch.no_grad()
def restore_params(params):
    for p in params:
        if hasattr(p, '_backup'):
            p.data = p._backup
            del p._backup

env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
policy = Policy().to(device)
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

reward_sum_running_avg = None

for it in range(100000):
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []

    for ep in range(10):
        obs, _ = env.reset()
        prev_obs = None

        for t in range(190000):
            frame = env.render()
            gif_frames.append(frame)
            if len(gif_frames) > 1200:
                gif_frames.pop(0)
            d_obs = policy.pre_process(obs, prev_obs)
            with torch.no_grad():
                action, action_prob = policy(d_obs)

            prev_obs = obs
            obs, reward, terminated, truncated, _ = env.step(policy.convert_action(action))
            done = terminated or truncated

            d_obs_history.append(d_obs)
            action_history.append(action)
            action_prob_history.append(action_prob)
            reward_history.append(reward)

            if done:
                reward_sum = sum(reward_history[-t:])
                reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum if reward_sum_running_avg else reward_sum
                print(f'Iteration {it}, Episode {ep} ({t} timesteps) - last_action: {action}, last_action_prob: {action_prob:.2f}, reward_sum: {reward_sum:.2f}, running_avg: {reward_sum_running_avg:.2f}')
                reward_sums.append(reward_sum)
                reward_running_avgs.append(reward_sum_running_avg)
                all_rewards.append(reward_history[:])
                break

    R = 0
    discounted_rewards = []
    for r in reward_history[::-1]:
        if r != 0:
            R = 0
        R = r + policy.gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

    for _ in range(5):
        n_batch = min(24576, len(action_history))
        idxs = random.sample(range(len(action_history)), n_batch)

        d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0).to(device)
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs]).to(device)
        action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs]).to(device)
        advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs]).to(device)

        opt.zero_grad()
        loss = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
        loss.backward()

        perturb_params(policy.parameters())
        opt.zero_grad()
        loss_sam = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
        loss_sam.backward()
        restore_params(policy.parameters())
        opt.step()

    if it % 5 == 0:
        torch.save(policy.state_dict(), 'params.ckpt')
        with open("logs/rewards.csv", "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Reward_Sum", "Running_Avg"])
            for i, (s, a) in enumerate(zip(reward_sums, reward_running_avgs)):
                writer.writerow([i, s, a])
        plt.figure()
        plt.plot(reward_sums, label="Reward Sum")
        plt.plot(reward_running_avgs, label="Running Avg")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Trend")
        plt.legend()
        plt.savefig("logs/reward_plot.png")
        plt.close()

        gif_path = f"logs/episode_{it}.gif"
        imageio.mimsave(gif_path, gif_frames[-1200:], duration=0.05)
        gif_frames.clear()
env.close()
