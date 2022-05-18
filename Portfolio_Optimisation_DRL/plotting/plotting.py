import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import figure
from pylab import rcParams
import glob
import os
from pathlib import Path
import configparser

params = { 
    'axes.labelsize': 12, 
    'font.size': 12, 
    'legend.fontsize': 12, 
    'xtick.labelsize': 12, 
    'ytick.labelsize': 12, 
    'figure.figsize': [8.5, 4.5]
}
FS=(4.9, 4.4)
rcParams.update(params)

def plot_data2(data_matrix, ax, lcolor, lline, label, p, lmarker=''):
    plot_only_each_n_point=1
    data_matrix=data_matrix.reshape((int(data_matrix.shape[0]*plot_only_each_n_point),int(data_matrix.shape[1]/plot_only_each_n_point)), order='F')
    x = np.arange(0, data_matrix.shape[1])
    x2= np.arange(0, data_matrix.shape[1])*p*plot_only_each_n_point
    lx = np.arange(0, data_matrix.shape[1], 1)
    lx2 = np.arange(0, data_matrix.shape[1])*p*plot_only_each_n_point
    ymean = np.mean(data_matrix, axis=0)
    ys=data_matrix
    ystd = np.std(ys, axis=0)
    ystderr = ystd / np.sqrt(len(ys))
    ax.fill_between(lx2*p, ymean - ystderr, ymean + ystderr, color=lcolor, alpha=.4)
    ax.fill_between(lx2*p, ymean - ystd,    ymean + ystd,    color=lcolor, alpha=.2)
    ax.plot(lx2*p, ymean[lx], linewidth=2, linestyle=lline, color=lcolor, label=label,
            marker=lmarker, markevery=1000, markersize=4.5)

def plot_from_file2(path, fig, ax, subplot, color, label, sign):
    results2 = []
    for directory in glob.iglob(path+'**/results/', recursive=True):
        print(directory)
        results = []
        for index in range(0, 78):
            for filename in glob.iglob(directory+'**_run'+str(index)+'.csv', recursive=False):
                subresults = pd.read_csv(filename, compression='gzip', sep=",")
                subresults = subresults['Reward']
                
                results.append(np.sum(subresults))
        results2.append(results)
    results2 = np.array(results2)
    plot_data2(results2, ax, color, sign, label, 1)
    return ax

def plot_from_file_paraworker2(path, fig, ax, subplot, color, label, sign):
    results2 = []
    nbworker=8
    for directory in glob.iglob(path+'**/results/', recursive=True):
        print(directory)
        results = []
        for index in range(0, 78): #original 30
            subresults2=[]
            for worker in range(0,nbworker):
                for filename in glob.iglob(directory+'**'+str(worker)+'_run'+str(index)+'.csv', recursive=False):
                    subresults = pd.read_csv(filename, compression='gzip', sep=",")
                    subresults = subresults['Reward']
                    subresults2.append(np.sum(subresults))
                    
            results.append(np.mean(subresults2))
        results2.append(results)
            
    results2 = np.array(results2)
    plot_data2(results2, ax, color, sign, label, 1)
    return ax

path_local = "C:/Users/kiero/Documents/MATH3092 Project 30685729/Portfolio_Optimisation_DRL/"
drawlist=['alle', 'sac', 'ppo', 'a2c']
my_labels= {"x1": "A2C", "x2": "PPO", "x4": "Random", "x6": "SAC"}

if "alle" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file_paraworker2(path_local + "portfoliov0_all/ppo_"+'**', fig, ax, 111, 'orange', my_labels["x2"], '-')
    plot_from_file_paraworker2(path_local + "portfoliov0_all/a2c_"+'**', fig, ax, 111, 'brown', my_labels["x1"], '-')
    plot_from_file_paraworker2(path_local + "portfoliov0_all/sac_"+'**', fig, ax, 111, 'magenta', my_labels["x6"], '-')
    plot_from_file2(path_local + "portfoliov0_all/randomm_"+'**', fig, ax, 111, 'dimgrey', my_labels["x4"], '--')

    ax.set_xlabel("Number of Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Learning performance")
    ax.legend()
    plt.savefig('learning_curves_episode.png')

if "ppo" in drawlist:
    fig, ax = plt.subplots()
    plot_from_file_paraworker2(path_local + "portfoliov0_all/ppo_"+'**', fig, ax, 111, 'orange', my_labels["x2"], '-')
    # plot_from_file_paraworker2(path_local + "portfoliov0_all/a2c_"+'**', fig, ax, 111, 'brown', my_labels["x1"], '-')
    # plot_from_file_paraworker2(path_local + "portfoliov0_all/sac_"+'**', fig, ax, 111, 'magenta', my_labels["x6"], '-')
    plot_from_file2(path_local + "portfoliov0_all/randomm_"+'**', fig, ax, 111, 'dimgrey', my_labels["x4"], '--')

    ax.set_xlabel("Number of Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Learning performance PPO vs Random")
    ax.legend()
    plt.savefig('learning_curves_episode_ppo.png')

if "a2c" in drawlist:
    fig, ax = plt.subplots()
    # plot_from_file_paraworker2(path_local + "portfoliov0_all/ppo_"+'**', fig, ax, 111, 'orange', my_labels["x2"], '-')
    plot_from_file_paraworker2(path_local + "portfoliov0_all/a2c_"+'**', fig, ax, 111, 'brown', my_labels["x1"], '-')
    # plot_from_file_paraworker2(path_local + "portfoliov0_all/sac_"+'**', fig, ax, 111, 'magenta', my_labels["x6"], '-')
    plot_from_file2(path_local + "portfoliov0_all/randomm_"+'**', fig, ax, 111, 'dimgrey', my_labels["x4"], '--')

    ax.set_xlabel("Number of Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Learning performance of A2C vs Random")
    ax.legend()
    plt.savefig('learning_curves_episode_a2c.png')

if "sac" in drawlist:
    fig, ax = plt.subplots()
    # plot_from_file_paraworker2(path_local + "portfoliov0_all/ppo_"+'**', fig, ax, 111, 'orange', my_labels["x2"], '-')
    # plot_from_file_paraworker2(path_local + "portfoliov0_all/a2c_"+'**', fig, ax, 111, 'brown', my_labels["x1"], '-')
    plot_from_file_paraworker2(path_local + "portfoliov0_all/sac_"+'**', fig, ax, 111, 'magenta', my_labels["x6"], '-')
    plot_from_file2(path_local + "portfoliov0_all/randomm_"+'**', fig, ax, 111, 'dimgrey', my_labels["x4"], '--')

    ax.set_xlabel("Number of Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Learning performance of SAC vs Random")
    ax.legend()
    plt.savefig('learning_curves_episode_sac.png')
