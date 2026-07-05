import argparse
import os
import time
from agents.dqn_agent import DQNAgent
from agents.qlearning_agent import QLearningAgent
from configs.config import DQNConfig, EnvConfig, QLearningConfig
from environments.snake_env import SnakeEnv
from utils.reproducibility import set_seed
from utils.stats import MovingAverage

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a saved RL agent checkpoint')
    p.add_argument('--checkpoint', required=True, help='Path to saved checkpoint')
    p.add_argument('--episodes', type=int, default=10, help='Number of eval episodes')
    p.add_argument('--grid', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--render', action='store_true', help='Print ASCII render each step')
    p.add_argument('--delay', type=float, default=0.05, help='Delay (s) between render frames')
    return p.parse_args()

def load_agent(checkpoint_path: str, obs_dim: int, n_actions: int):
    if checkpoint_path.endswith('.pt'):
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(checkpoint_path)
        return agent
    elif checkpoint_path.endswith('.pkl'):
        agent = QLearningAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(checkpoint_path)
        return agent
    else:
        raise ValueError(f'Unknown checkpoint format: {checkpoint_path}')

def run_eval(agent, env: SnakeEnv, n_episodes: int, render: bool, delay: float):
    reward_ma = MovingAverage(window=n_episodes)
    all_scores = []
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        while True:
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                frame = env._render_ansi()
                print(f'Episode {ep}/{n_episodes}')
                print(frame)
                time.sleep(delay)
            if terminated or truncated:
                break
        reward_ma.push(total_reward)
        all_scores.append(info['score'])
        print(f'  ep {ep:>3} | reward {total_reward:>7.2f} | score {info['score']:>3} | steps {info['steps']:>4}')
    print(f'\n{'─' * 50}')
    print(f'  Mean reward : {reward_ma.mean:.2f}')
    print(f'  Mean score  : {sum(all_scores) / len(all_scores):.2f}')
    print(f'  Max score   : {max(all_scores)}')
    print(f'  Min score   : {min(all_scores)}')

def main():
    args = parse_args()
    set_seed(args.seed)
    render_mode = 'ansi' if args.render else 'none'
    env = SnakeEnv(grid_size=args.grid, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f'\nLoading checkpoint: {args.checkpoint}')
    agent = load_agent(args.checkpoint, obs_dim, n_actions)
    print(f'Agent: {agent.__class__.__name__}')
    print(f'Episodes: {args.episodes}  |  Grid: {args.grid}×{args.grid}')
    print(f'{'─' * 50}')
    run_eval(agent, env, args.episodes, args.render, args.delay)
if __name__ == '__main__':
    main()
