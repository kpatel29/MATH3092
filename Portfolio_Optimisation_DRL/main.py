import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO, A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.portfolio_trading_env import PortfolioTradingEnv as Env

def make_env(rank):
    """
    Utility function for multiprocessed env.

    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = Env(portfolio_stocks=["AAPL", "AMZN", "NFLX", "GOOGL"], initial_cash=100000, start_day="2012-01-01", 
                  end_day="2016-12-30", out_csv_name='results/rewards{}'.format(rank))
        return env
    
    return _init

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Args for Env""")
    prs.add_argument("-method", dest="method", type=str, default='ppo', required=False, help="which file to run.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-gamma", dest="gamma", type=float, default=0.99, required=False, help="discount factor.\n")
    prs.add_argument("-st", dest="steps", type=int, default=2048, required=False, help="n steps.\n")
    prs.add_argument("-batch", dest="batch_size", type=int, default=128, required=False, help="batch_size. \n")
    prs.add_argument("-cr", dest="clip_range", type=float, default=0.2, required=False, help="clip_range. \n")
    prs.add_argument('-vfc', dest="vf_coef", type=float, default=0.5, required=False, help="vf coef. \n")
    prs.add_argument('-efc', dest="ent_coef", type=float, default=0.0, required=False, help="ent coef. \n")
    prs.add_argument('-maxgrad', dest="max_grad_norm", type=float, default=0.5, required=False, help="max grad norm \n")
    prs.add_argument('-window', dest="loop_window", type=int, default=6, required=False, help="lopp window \n")
    args = prs.parse_args()

    num_cpu = 8  # Number of processes to use
    env = Env(portfolio_stocks=["AAPL", "AMZN", "NFLX", "GOOGL"], initial_cash=100000, start_day="2012-01-01", 
                  end_day="2016-12-30", out_csv_name='results/rewards')
    
    if args.method == 'randomm': #baseline
        env.reset()
        for i in range(100000):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            if done:
                env.reset()
    else:
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        if args.method == 'ppo':
            model = PPO('MlpPolicy', env, gamma=0.95, verbose=1)
            #number of frames in its completion
            model.learn(total_timesteps=800000) 
            
        elif args.method == 'a2c':
            
            model = A2C('MlpPolicy', env, gamma=0.95, verbose=0)
            
            model.learn(total_timesteps=800000)
        elif args.method == "sac":
            model = SAC('MlpPolicy', env, gamma=0.95, verbose=0)
            
            model.learn(total_timesteps=800000)
        else:
            print('Invalid Algorithm')
