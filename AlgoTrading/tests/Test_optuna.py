import pandas as pd
import numpy as np
import pickle
import optuna
from optuna.visualization import plot_optimization_history
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from mlfinlab.online_portfolio_selection import *
import math

quote           = 'BTC'
pair            = 'ADABTC'
timeframe       = '1h'
ClosesOrReturns = ''

# Read in the data.
print("_________ " + pair + " _________")
filename = 'Historical_data/' + quote + '/' + pair + '_' + timeframe + '_log'
df = pd.read_csv(filename, sep='\t', parse_dates=True, index_col='time')
df = df[['time', 'close']]



def _reward(self):
    length  = min(self.current_step, self.reward_len)
    returns = np.diff(self.net_worths)[-length:]

    if self.reward_func == 'sortino':
        reward = sortino_ratio(returns)
    elif self.reward_func == 'calmar':
        reward = calmar_ratio(returns)
    elif self.reward_func == 'omega':
        reward = omega_ratio(returns)
    else:
        reward = np.mean(returns)

    return reward if abs(reward) != math.inf and not np.isnan(reward) else 0


def optimize_envs(trial):
    return {
        'reward_len'          : int(trial.suggest_loguniform('reward_len', 1, 200)),
        'forecast_len'        : int(trial.suggest_loguniform('forecast_len', 1, 200)),
        'confidence_interval' : trial.suggest_uniform('confidence_interval', 0.7, 0.99),
    }

def optimize_ppo2(trial):
    return {
        'n_steps'       : int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma'         : trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef'      : trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange'     : trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs'    : int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam'           : trial.suggest_uniform('lam', 0.8, 1.)
    }


def objective_fn(trial):
    env_params   = optimize_envs(trial)
    agent_params = optimize_ppo2(trial)

    train_env, validation_env = initialize_envs(**env_params)
    model = PPO2(MlpLstmPolicy, train_env, **agent_params)

    model.learn(len(train_env.df))

    rewards, done = [], False

    obs = validation_env.reset()
    for i in range(len(validation_env.df)):
        action, _ = model.predict(obs)
        obs, reward, done, _ = validation_env.step(action)
        rewards += reward

    return -np.mean(rewards)


def optimize(n_trials = 100, n_jobs = 4):
    study = optuna.create_study(study_name='optimize_profit', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)


if __name__ == "__main__":
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    study = optuna.create_study()
    study.optimize(objective_fn, n_trials=50)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # # We can continue the optimization as follows.
    # print("Running 20 additional trials...")
    # study.optimize(objective, n_trials=20)
    # print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # optuna.visualization.plot_optimization_history(study).show()          # Visualizing the Optimization History
    # optuna.visualization.plot_contour(study, params=['x', 'y']).show()    # Visualizing Hyperparameter Relationships
    # optuna.visualization.plot_parallel_coordinate(study).show()           # Visualizing High-dimensional Parameter Relationships
    optuna.visualization.plot_slice(study).show()                           # Visualizing Individual Hyperparameters



