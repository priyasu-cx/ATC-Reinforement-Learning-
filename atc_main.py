import datetime
import os
import uuid
from multiprocessing import freeze_support

import gym
import yaml
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.sac.policies import MlpPolicy
import stable_baselines3.sac.policies as sacpolicies

# from stable_baselines3.common.schedules import LinearSchedule
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecVideoRecorder,
)
import tensorflow as tf
import numpy as np
from typing import Callable
import time

# noinspection PyUnresolvedReferences
import envs.atc.atc_gym


def LinearSchedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class ModelFactory:
    hyperparams: dict

    def build(self, env, log_dir):
        pass


def learn(
    model_factory: ModelFactory,
    multiprocess: bool = True,
    time_steps: int = int(1e6),
    record_video: bool = True,
):
    def callback(locals_, globals_):
        locals_["fps"] = 0
        locals_["ep_infos"] = []
        self_ = locals_["self"]
        locals_["writer"] = tf.summary.create_file_writer("/tmp/tf2_summary_example")

        mean_actions = np.mean(self_.env.get_attr("actions_per_timestep"))
        mean_actions_tf = tf.summary.scalar(
            name="simulation/mean_actions", data=mean_actions
        )
        winning_ratio = np.mean(self_.env.get_attr("winning_ratio"))
        winning_ratio_tf = tf.summary.scalar(
            name="simulation/winning_ratio", data=winning_ratio
        )

        with locals_["writer"].as_default():
            tf.summary.scalar(
                "simulation/mean_actions", mean_actions_tf, step=self_.num_timesteps
            )
            tf.summary.scalar(
                "simulation/winning_ratio", winning_ratio_tf, step=self_.num_timesteps
            )

        if isinstance(model_factory, PPO2ModelFactory):

            fps = tf.summary.scalar(name="simulation/fps", data=locals_["fps"])
            mean_length = np.mean([info["l"] for info in locals_["ep_infos"]])

            mean_length_tf = tf.summary.scalar(
                name="simulation/mean_episode_length", data=mean_length
            )

            with locals_["writer"].as_default():
                tf.summary.scalar("simulation/fps", fps, step=self_.num_timesteps)
                tf.summary.scalar(
                    "simulation/mean_episode_length",
                    mean_length_tf,
                    step=self_.num_timesteps,
                )
        return True

    def video_trigger(step):
        # allow warm-up for video recording
        if not record_video or step < time_steps / 3:
            return False

        return step % (int(time_steps / 8)) == 0

    log_dir = "../logs/%s/" % datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    log_dir_tensorboard = "../logs/tensorboard/"
    print("Tensorboard log directory: %s" % os.path.abspath(log_dir_tensorboard))

    model_dir = os.path.join(log_dir, "model")

    os.makedirs(model_dir, exist_ok=True)

    video_dir = os.path.join(log_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    def make_env():
        log_dir_single = "%s/%s/" % (log_dir, uuid.uuid4())
        env = gym.make("AtcEnv-v0")
        os.makedirs(log_dir_single, exist_ok=True)
        env = Monitor(env, log_dir_single, allow_early_resets=True)
        return env

    # No of environments to run in parallel
    n_envs = 8

    if multiprocess:
        env = SubprocVecEnv([lambda: make_env() for i in range(n_envs)])
    else:
        env = DummyVecEnv([lambda: make_env()])

    if record_video:
        env = VecVideoRecorder(env, video_dir, video_trigger, video_length=2000)

    # Build or Load model
    # model = model_factory.build(env, log_dir_tensorboard)
    model = PPO.load("PPO2_atc_gym_1", env=env, print_system_info=True)

    yaml.dump(
        model_factory.hyperparams,
        open(os.path.join(model_dir, "hyperparams.yml"), "w+"),
    )

    # model = ACKTR(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=time_steps, callback=callback)

    model.save("%s/PPO2_atc_gym" % model_dir)

    # render trained model actions on screen and to file
    eval_observations_file = open(os.path.join(model_dir, "evaluation.csv"), "a+")
    new_env = gym.make("AtcEnv-v0")
    obs = new_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = new_env.step(action)
        original_state = info["original_state"]
        eval_observations_file.write(
            "%.2f, %.2f, %.0f, %.1f\n"
            % (
                original_state[0],
                original_state[1],
                original_state[2],
                original_state[3],
            )
        )
        new_env.render()
        if done:
            if rewards < 0:
                print("LOST", rewards)
            else:
                print("WON", rewards)
            time.sleep(5)
            obs = new_env.reset()


class PPO2ModelFactory(ModelFactory):
    def __init__(self):
        self.hyperparams = {
            "n_steps": 1024,
            "batch_size": 32,
            "clip_range": 0.4,
            "gamma": 0.996,
            "gae_lambda": 0.95,
            # "learning_rate": LinearSchedule(1.0, initial_p=0.0002, final_p=0.001).value,
            "learning_rate": LinearSchedule(initial_value=0.0002),
            "n_epochs": 4,
            "ent_coef": 0.002,
        }

    def build(self, env, log_dir):
        return PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **self.hyperparams
        )


class SACModelFactory(ModelFactory):
    def __init__(self):
        self.hyperparams = {
            "learning_rate": 3e-4,
            "buffer_size": 1000000,
            "batch_size": 256,
            "ent_coef": "auto",
            "gamma": 0.99,
            "train_freq": 1,
            "tau": 0.005,
            "gradient_steps": 1,
            "learning_starts": 1000,
        }

    def build(self, env, log_dir):
        return SAC(
            sacpolicies.MlpPolicy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **self.hyperparams
        )


if __name__ == "__main__":
    freeze_support()
    learn(
        PPO2ModelFactory(),
        time_steps=int(1000000),
        multiprocess=True,
        record_video=False,
    )
