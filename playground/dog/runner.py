"""Runs training and evaluation loop for the Dog quadruped robot."""

import os
import argparse
import functools

# Force TensorFlow to use CPU only (prevent OOM when GPU is used by JAX)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
# Re-enable GPU for JAX
del os.environ["CUDA_VISIBLE_DEVICES"]

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.dog import joystick, standing, stair_climb

from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper


class DogRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "standing": (standing, standing.Standing),
            "stair_climb": (stair_climb, stair_climb.StairClimb),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )
        self.restore_checkpoint_path = args.restore_checkpoint_path
        if args.num_envs is not None:
            self.num_envs_override = args.num_envs
        else:
            self.num_envs_override = None
        print(f"Observation size: {self.obs_size}")

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )
        self.ppo_training_params = dict(self.ppo_params)

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks

        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        if self.num_envs_override is not None:
            self.ppo_training_params["num_envs"] = self.num_envs_override
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dog Quadruped Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override number of parallel envs (reduce if OOM, default 8192)",
    )
    args = parser.parse_args()

    runner = DogRunner(args)

    runner.train()


if __name__ == "__main__":
    main()
