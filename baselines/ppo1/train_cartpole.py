#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
import gym
import logging
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser


def callback(lcl, glb):
    # stop training if mean episode reward exceeds 199
    is_solved = np.mean(lcl['rewbuffer']) >= 199
    return is_solved


def train(env_id, num_timesteps, seed, save_model_with_prefix, restore_model_from_file):
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps),
        timesteps_per_actorbatch=2048,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear',
        callback=callback,
        save_model_with_prefix=save_model_with_prefix,
        restore_model_from_file=restore_model_from_file
        )
    env.close()
    saver = tf.train.Saver()
    saver.save(sess, '/tmp/model')


def main():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--save_model_with_prefix', help='Specify a prefix name to save the model with after every 500 iters. Note that this will generate multiple files (*.data, *.index, *.meta and checkpoint) with the same prefix', default='')
    parser.add_argument('--restore_model_from_file', help='Specify the absolute path to the model file including the file name upto .model (without the .data-00000-of-00001 suffix). make sure the *.index and the *.meta files for the model exists in the specified location as well', default='')
    args = parser.parse_args()
    train(args.env, num_timesteps=1e6, seed=0, save_model_with_prefix=args.save_model_with_prefix, restore_model_from_file=args.restore_model_from_file)


if __name__ == "__main__":
    main()
