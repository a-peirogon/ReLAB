import argparse
import os
import random
import sys
import time
import numpy as np
from agents.dqn_agent import DQNAgent
from agents.qlearning_agent import QLearningAgent
from environments.snake_env import SnakeEnv

def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def parse_args():
    p = argparse.ArgumentParser(description='Visualización en vivo del agente Snake')
    p.add_argument('--checkpoint', type=str, default=None, help='Ruta al checkpoint (.pt para DQN, .pkl para Q-Learning)')
    p.add_argument('--agent', choices=['dqn', 'qlearning'], default='dqn')
    p.add_argument('--grid', type=int, default=18, help='Tamaño del grid')
    p.add_argument('--cell', type=int, default=20, help='Píxeles por celda')
    p.add_argument('--fps', type=int, default=10, help='Fotogramas por segundo')
    p.add_argument('--episodes', type=int, default=0, help='0 = solo demo, N = entrenar N episodios con visualización')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--explore', action='store_true', help='Usar exploración epsilon (solo demo)')
    return p.parse_args()

def load_agent(path: str, obs_dim: int, n_actions: int):
    if path.endswith('.pt'):
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(path)
        print(f'  Cargado DQNAgent desde {path}')
    elif path.endswith('.pkl'):
        agent = QLearningAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(path)
        print(f'  Cargado QLearningAgent desde {path}  (tabla: {agent.table_size} estados)')
    else:
        raise ValueError(f'Extensión de checkpoint desconocida: {path}')
    return agent

def get_dqn_activations(agent: DQNAgent, obs: np.ndarray) -> list[np.ndarray]:
    import torch
    agent.policy_net.eval()
    activations = []
    x = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
    activations.append(obs.copy())
    with torch.no_grad():
        for layer in agent.policy_net.net:
            x = layer(x)
            import torch.nn as nn
            if isinstance(layer, (nn.ReLU, nn.Tanh)):
                activations.append(x.squeeze(0).cpu().numpy())
        if len(activations) < 4:
            activations.append(x.squeeze(0).cpu().numpy())
    return activations

def get_qlearning_activations(agent: QLearningAgent, obs: np.ndarray) -> list[np.ndarray]:
    key = agent._obs_to_key(obs)
    q_vals = agent.q_table[key]
    return [obs.copy(), q_vals.copy()]

def main():
    args = parse_args()
    set_seed(args.seed)
    try:
        import pygame
    except ImportError:
        print('ERROR: pygame no instalado. Ejecuta:  pip install pygame')
        sys.exit(1)
    from environments.pygame_renderer import PygameRenderer
    env = SnakeEnv(grid_size=args.grid, step_penalty=-0.01, max_steps=2000, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent = load_agent(args.checkpoint, obs_dim, n_actions)
    else:
        if args.checkpoint:
            print(f'  Checkpoint no encontrado: {args.checkpoint}')
        print('  Usando agente DQN sin entrenar (pesos aleatorios)')
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=128)
    if isinstance(agent, DQNAgent):
        hidden = agent.policy_net.net[0].out_features
        nn_layers = [obs_dim, hidden, hidden, n_actions]
    else:
        nn_layers = [obs_dim, n_actions]
    renderer = PygameRenderer(grid_size=args.grid, cell_size=args.cell, nn_layers=nn_layers, fps=args.fps)
    train_mode = args.episodes > 0
    episode = 0
    max_ep = args.episodes if train_mode else 10000
    print(f'\n  Snake en vivo  |  grid={args.grid}×{args.grid}  fps={args.fps}')
    print(f'  Modo: {('ENTRENAMIENTO' if train_mode else 'DEMO')}  |  Q para salir\n')
    try:
        while episode < max_ep:
            obs, info = env.reset()
            episode += 1
            running = True
            step = 0
            while running:
                if isinstance(agent, DQNAgent):
                    acts = get_dqn_activations(agent, obs)
                elif isinstance(agent, QLearningAgent):
                    acts = get_qlearning_activations(agent, obs)
                else:
                    acts = None
                epsilon = getattr(agent, 'epsilon', 0.0)
                alive = renderer.render(snake=list(env.snake), food=env.food, obs=obs, score=env.score, activations=acts, episode=episode, step_count=step, epsilon=epsilon)
                if not alive:
                    raise KeyboardInterrupt
                renderer.tick()
                explore = train_mode or args.explore
                action = agent.select_action(obs, explore=explore)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if train_mode:
                    agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
                step += 1
                running = not done
            if train_mode:
                agent.on_episode_end()
                if episode % 50 == 0:
                    eps_str = f'  ε={agent.epsilon:.3f}' if hasattr(agent, 'epsilon') else ''
                    print(f'  ep {episode:>4}  score={info['score']:>3}  steps={step:>4}{eps_str}')
    except KeyboardInterrupt:
        print('\n  Saliendo...')
    finally:
        renderer.close()
        env.close()
if __name__ == '__main__':
    main()
