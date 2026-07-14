"""
run_watch_gridworld.py — Entrenamiento EN VIVO de Q-learning sobre GridWorld.

A diferencia de los demás run_watch_*, este script entrena mientras
renderiza: la razón de ser de GridWorld es ver el Q-table pasar de todo
negro a un mapa de calor con flechas de política, propagándose desde la
meta hacia atrás, episodio a episodio.

Uso:
    python run_watch_gridworld.py                        # maze, por defecto
    python run_watch_gridworld.py --layout cliff          # Cliff Walking (Sutton & Barto)
    python run_watch_gridworld.py --layout empty --size 8
    python run_watch_gridworld.py --render_every 5        # renderizar 1 de cada 5 episodios (más rápido)
    python run_watch_gridworld.py --episodes 500 --fps 60
"""

import argparse
import sys

from agents.qlearning_agent import QLearningAgent
from environments.gridworld_env import GridWorldEnv
from utils.reproducibility import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento en vivo — GridWorld")
    p.add_argument("--layout",       choices=["empty", "maze", "cliff"], default="maze")
    p.add_argument("--size",         type=int,   default=10, help="tamaño de grilla (solo layout=empty)")
    p.add_argument("--episodes",     type=int,   default=300)
    p.add_argument("--render_every", type=int,   default=1, help="renderizar 1 de cada N episodios")
    p.add_argument("--fps",          type=int,   default=30)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--gamma",        type=float, default=0.95)
    p.add_argument("--epsilon_decay", type=float, default=0.97)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--checkpoint",   type=str,   default=None, help="guardar Q-table entrenada aquí")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("ERROR: pygame no instalado. Ejecuta:  pip install pygame")
        sys.exit(1)

    from environments.gridworld_renderer import GridWorldRenderer

    env = GridWorldEnv(layout=args.layout, size=args.size, seed=args.seed)
    agent = QLearningAgent(
        obs_dim=2, n_actions=env.action_space.n,
        lr=args.lr, gamma=args.gamma, epsilon_decay=args.epsilon_decay,
    )
    renderer = GridWorldRenderer(n_rows=env.n_rows, n_cols=env.n_cols, fps=args.fps)

    print(f"\n  GridWorld en vivo  |  layout={args.layout}  ({env.n_rows}x{env.n_cols})")
    print("  Q para salir\n")

    solved_count = 0
    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            render_this_ep = (episode % args.render_every == 0) or episode == 1
            done = False
            step = 0

            while not done:
                action = agent.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, next_obs, terminated or truncated)
                obs = next_obs
                done = terminated or truncated
                step += 1

                if render_this_ep:
                    alive = renderer.render(
                        agent_pos=env.agent_pos,
                        goal=env.goal,
                        walls=env.walls,
                        pits=env.pits,
                        q_table=agent.q_table,
                        episode=episode,
                        step_count=step,
                        epsilon=agent.epsilon,
                        episodes_solved=solved_count,
                    )
                    if not alive:
                        raise KeyboardInterrupt
                    renderer.tick()

            agent.on_episode_end()
            if info.get("score"):
                solved_count += 1

            if episode % 20 == 0 or episode == 1:
                outcome = "GOAL" if info.get("score") else ("PIT" if info.get("fell_in_pit") else "TIMEOUT")
                print(f"  ep {episode:>4}/{args.episodes}  {outcome:<8}  steps={step:<4}  "
                      f"eps={agent.epsilon:.3f}  solved={solved_count}  states={agent.table_size}")

    except KeyboardInterrupt:
        print("\n  Saliendo...")
    finally:
        renderer.close()
        env.close()

    if args.checkpoint:
        agent.save(args.checkpoint)
        print(f"\n  Q-table guardada en {args.checkpoint}")

    print(f"\n  Total resuelto: {solved_count}/{args.episodes}  ({100*solved_count/args.episodes:.1f}%)")


if __name__ == "__main__":
    main()
