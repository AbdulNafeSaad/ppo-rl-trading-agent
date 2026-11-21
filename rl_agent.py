import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for clean terminal execution

import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import STOCKS_GOOGL
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load Real Historical Market Data (GOOGL Daily OHLCV)
df = STOCKS_GOOGL.copy()
split_idx = int(len(df) * 0.8)  # 80% Training, 20% Out-of-Sample Testing
window_size = 10

# 2. Initialize Chronological Train & Out-of-Sample Test Environments
def make_train_env():
    return gym.make("stocks-v0", df=df, frame_bound=(window_size, split_idx), window_size=window_size)

def make_test_env():
    return gym.make("stocks-v0", df=df, frame_bound=(split_idx, len(df)), window_size=window_size)

train_env = DummyVecEnv([make_train_env])

# 3. Train PPO Agent on Historical Training Window
model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003, seed=42)
print(f"Training PPO agent on {split_idx - window_size} historical trading days...")
model.learn(total_timesteps=100000)
model.save("d:/DataScienceProjects_IA/ppo_autonomous_agent")

# 4. Out-of-Sample Evaluation on Unseen Test Window
test_env = make_test_env()
obs, info = test_env.reset(seed=42)
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

# 5. Compute Baseline & Quantitative Risk Metrics
test_prices = df["Close"].iloc[split_idx:len(df)]
buy_and_hold_return = (test_prices.iloc[-1] / test_prices.iloc[0] - 1.0) * 100
ppo_return = (float(info["total_profit"]) - 1.0) * 100

profits = np.array(test_env.unwrapped.history["total_profit"], dtype=float)
step_returns = np.diff(profits) / profits[:-1]
sharpe_ratio = (np.mean(step_returns) / (np.std(step_returns) + 1e-8)) * np.sqrt(252)
running_max = np.maximum.accumulate(profits)
max_drawdown = np.min((profits - running_max) / running_max) * 100

print("\n--- Out-of-Sample Backtest Results ---")
print(f"Test Window                 : {df.index[split_idx].date()} to {df.index[-1].date()} ({len(test_prices)} days)")
print(f"PPO Agent Cumulative Return : {ppo_return:.2f}%")
print(f"Buy & Hold Baseline Return  : {buy_and_hold_return:.2f}%")
print(f"Annualized Sharpe Ratio     : {sharpe_ratio:.2f}")
print(f"Maximum Drawdown            : {max_drawdown:.2f}%")
print(f"Final Execution Info        : {info}")

# 6. Export Policy Entry/Exit Chart for GitHub README
plt.figure(figsize=(14, 6))
test_env.unwrapped.render_all()
plt.title(f"PPO Out-of-Sample Trading Policy (Return: {ppo_return:.2f}% vs Buy & Hold: {buy_and_hold_return:.2f}%)")
plt.tight_layout()
plt.savefig("d:/DataScienceProjects_IA/trading_actions.png", dpi=300)
print("Saved trade visualization to d:/DataScienceProjects_IA/trading_actions.png")