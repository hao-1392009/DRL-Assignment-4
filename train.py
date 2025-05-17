import argparse
import json
import pathlib
import os
import sys
import logging
import random

import gymnasium as gym
import numpy as np
import torch

import dmc
import sac
from replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%m-%d %H:%M:%S")


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config")
    parser.add_argument("--env_name", help="must be Pendulum-v1, cartpole-balance, or humanoid-walk")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gamma", type=float, help="discount factor")
    parser.add_argument("--tau", type=float, help="soft target update")
    parser.add_argument("--alpha_init", type=float)
    parser.add_argument("--replay_buffer_size", type=int)
    parser.add_argument("--steps_before_training", type=int)
    parser.add_argument("--steps_per_online_update", type=int)
    parser.add_argument("--total_episodes", type=int)
    parser.add_argument("--episodes_per_log", type=int)
    parser.add_argument("--episodes_per_save", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--resume_from_checkpoint", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)

    return parser

def fix_random_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


def main():
    parser = get_arg_parser()

    # precedence: command line arguments > config file arguments > default arguments
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as file:
            vars(args).update(json.load(file))
        parser.parse_args(namespace=args)

    output_dir = pathlib.Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        handlers=[stdout_handler, file_handler],
        force=True
    )

    logger.info(f"Training config: {vars(args)}")


    if args.env_name == "Pendulum-v1":
        env = gym.make("Pendulum-v1", render_mode='rgb_array')
    elif args.env_name == "cartpole-balance":
        env = dmc.make_dmc_env("cartpole-balance", np.random.randint(0, 1000000),
                               flatten=True, use_pixels=False)
    elif args.env_name == "humanoid-walk":
        env = dmc.make_dmc_env("humanoid-walk", np.random.randint(0, 1000000),
                               flatten=True, use_pixels=False)
    else:
        raise ValueError("Invalid env_name")

    fix_random_seed(args.seed, env)

    if args.resume_from_checkpoint:
        checkpoint_dir = output_dir / f"checkpoint-{args.resume_from_checkpoint}"
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        replay_buffer = torch.load(checkpoint_dir / "replay_buffer.pt")
        logger.info(f"Successfully loaded replay buffer from {checkpoint_dir}")
    else:
        checkpoint_dir = None
        replay_buffer = ReplayBuffer(args.replay_buffer_size, args.device)

    agent = sac.SAC(env.observation_space.shape[0], env.action_space.shape[0],
                    env.action_space.high, args, checkpoint_dir)


    if not args.resume_from_checkpoint:
        logger.info(f"Start collecting {args.steps_before_training} transitions")
        state, info = env.reset()
        for _ in range(args.steps_before_training):
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if len(replay_buffer) == replay_buffer.capacity // 2:
                logger.debug("replay buffer half full")
            elif len(replay_buffer) == replay_buffer.capacity - 1:
                logger.debug("replay buffer full")
            replay_buffer.add(state, action, reward, next_state, done)

            if done:
                state, info = env.reset()
            else:
                state = next_state


    reward_history = []
    avg_reward_history = []
    step_counter = 0

    logger.info("***** Start training *****")
    for episode in range(args.resume_from_checkpoint + 1,
                         args.resume_from_checkpoint + args.total_episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if len(replay_buffer) == replay_buffer.capacity // 2:
                logger.debug("replay buffer half full")
            elif len(replay_buffer) == replay_buffer.capacity - 1:
                logger.debug("replay buffer full")
            replay_buffer.add(state, action, reward, next_state, done)

            step_counter += 1
            if step_counter % args.steps_per_online_update == 0:
                agent.update_online(*replay_buffer.sample(args.batch_size))

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        if episode % args.episodes_per_log == 0:
            avg_reward = np.mean(reward_history)
            logger.info(f"Episode: {episode}, Average Reward: {avg_reward}")
            reward_history = []

            avg_reward_history.append((episode, avg_reward))

            agent.log()

        if episode % args.episodes_per_save == 0:
            checkpoint_dir = output_dir / f"checkpoint-{episode}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            logger.info(f"Saving checkpoint to {checkpoint_dir}")
            agent.save(checkpoint_dir)
            torch.save(replay_buffer, checkpoint_dir / "replay_buffer.pt")

            with open(output_dir / "avg_reward_history.txt", "a") as file:
                for epi, avg_reward in avg_reward_history:
                    file.write(f"{epi} {avg_reward}\n")
            avg_reward_history = []

            logger.info(f"Successfully saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()
