import argparse
import os
import sys
from agents.dqn_agent import DQNAgent
from agents.dwa_agent import DWAAgent
from environments.robot_nav_env import RobotNavEnv
from utils.reproducibility import set_seed

def parse_args():
    p = argparse.ArgumentParser(description='Visualización en vivo — Robot Navigation')
    p.add_argument('--agent', choices=['dwa', 'dqn'], default='dwa')
    p.add_argument('--checkpoint', type=str, default=None, help='Checkpoint .pt para DQN (opcional; sin él usa pesos aleatorios)')
    p.add_argument('--obstacles', type=int, default=8)
    p.add_argument('--obstacle_speed', type=float, default=0.8)
    p.add_argument('--world', type=float, default=12.0)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--episodes', type=int, default=0, help='0 = demo infinita, N = N episodios y termina')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def build_agent(args, obs_dim: int, n_actions: int):
    if args.agent == 'dwa':
        return DWAAgent(obs_dim=obs_dim, n_actions=n_actions, n_obstacles=args.obstacles, world_size=args.world, obstacle_speed=args.obstacle_speed)
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=128)
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f'  Cargado DQNAgent desde {args.checkpoint}')
    else:
        if args.checkpoint:
            print(f'  Checkpoint no encontrado: {args.checkpoint}')
        print('  Usando DQN sin entrenar (pesos aleatorios) — entrénalo con:')
        print('    python run_train.py --env robot_nav')
    return agent

def main():
    args = parse_args()
    set_seed(args.seed)
    try:
        import pygame
    except ImportError:
        print('ERROR: pygame no instalado. Ejecuta:  pip install pygame')
        sys.exit(1)
    from environments.robot_nav_renderer import RobotNavRenderer
    env = RobotNavEnv(world_size=args.world, n_obstacles=args.obstacles, obstacle_speed=args.obstacle_speed, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = build_agent(args, obs_dim, n_actions)
    renderer = RobotNavRenderer(world_size=args.world, fps=args.fps)
    episode = 0
    max_episodes = args.episodes if args.episodes > 0 else 10000
    total_reached = 0
    print(f'\n  Robot Navigation en vivo  |  agente={args.agent}  obstacles={args.obstacles}')
    print('  Q para salir\n')
    try:
        while episode < max_episodes:
            obs, info = env.reset()
            episode += 1
            running = True
            step = 0
            while running:
                action = agent.select_action(obs, explore=False)
                candidates = getattr(agent, 'last_candidates', None)
                best_idx = getattr(agent, 'last_best_idx', 0)
                alive = renderer.render(robot_pose=(env.x, env.y, env.theta), goal=env.goal, obstacles=env.obstacles, episode=episode, step_count=step, agent_name=args.agent.upper(), score=total_reached, candidates=candidates, best_idx=best_idx)
                if not alive:
                    raise KeyboardInterrupt
                renderer.tick()
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                running = not (terminated or truncated)
            if info.get('score'):
                total_reached += 1
            outcome = 'GOAL' if info.get('score') else 'COLLISION' if info.get('collided') else 'TIMEOUT'
            print(f'  ep {episode:>4}  steps={step:>4}  {outcome:<10}  reached_total={total_reached}')
    except KeyboardInterrupt:
        print('\n  Saliendo...')
    finally:
        renderer.close()
        env.close()
if __name__ == '__main__':
    main()
