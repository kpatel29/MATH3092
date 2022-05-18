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
    'figure.figsize': [7.8, 2.8]
}
FS=(4.9, 4.4)
rcParams.update(params)

def plot_from_file(path, fig, subplot, color, sign):
    results2 = []
    for directory in glob.iglob(path+'**/results/', recursive=True):
        print(directory)
        results = []
        for index in range(70,78):
            for filename in glob.iglob(directory+'**_run'+str(index)+'.csv', recursive=False):
                subresults = pd.read_csv(filename, compression='gzip', sep=",")
                subresults = [subresults['Reward']]

            results.append(np.array(subresults).mean(axis=1))
        results = np.asarray(results)
        results = results.mean(axis=0)
        results2.append(results)
    results2 = np.array(results2)
    return results2

def plot_from_species(path, fig, subplot, color, sign):
    
    results2 = []
    nbworker=8
    for directory in glob.iglob(path+'**/results/', recursive=True):
        print(directory)
        results = []
        for index in range(70,78):
            subresults2=[]
            for worker in range(0,nbworker):
                for filename in glob.iglob(directory+'**rewards'+str(worker)+'_run'+str(index)+'.csv', recursive=False):
                    subresults = pd.read_csv(filename, compression='gzip', sep=",")
                    subresults = [subresults['Reward']]
                    
                    subresults2.append(np.array(subresults).mean(axis=1))        
            subresults2 = np.array(subresults2).mean(axis=0)
            results.append(subresults2)
            
        results = np.asarray(results)
        results = results.mean(axis=0)
        results2.append(results)

    results2 = np.array(results2)
    return results2

def plot_from_file_assets(path, fig, subplot, color, sign):
    results2 = []
    for directory in glob.iglob(path+'**/results/', recursive=True):
        print(directory)
        results = []
        for index in range(70,73):
            for filename in glob.iglob(directory+'**_run'+str(index)+'.csv', recursive=False):
                subresults = pd.read_csv(filename, compression='gzip', sep=",")
                subresults = [subresults["assets"]]

            results.append(np.array(subresults).mean(axis=1))
        results = np.asarray(results)
        results = results.mean(axis=0)
        results2.append(results)
    results2 = np.array(results2)
    return results2

def plot_from_species_assets(path, fig, subplot, color, sign):
    
    results2 = []
    nbworker=8
    for directory in glob.iglob(path+'**/results/', recursive=True):
        print(directory)
        results = []
        for index in range(70,73):
            subresults2=[]
            for worker in range(0,nbworker):
                for filename in glob.iglob(directory+'**rewards'+str(worker)+'_run'+str(index)+'.csv', recursive=False):
                    subresults = pd.read_csv(filename, compression='gzip', sep=",")
                    subresults = [subresults["assets"]]
                    
                    subresults2.append(np.array(subresults).mean(axis=1))        
            subresults2 = np.array(subresults2).mean(axis=0)
            results.append(subresults2)
            
        results = np.asarray(results)
        results = results.mean(axis=0)
        results2.append(results)

    results2 = np.array(results2)
    return results2


path_local = "C:/Users/kiero/Documents/MATH3092 Project 30685729/Portfolio_Optimisation_DRL/"
drawlist=['reward', 'assets']
 
if "reward" in drawlist:

    fig = figure()
    random = plot_from_file(path_local + "portfoliov0_all/randomm_" +'**', fig, 111, 'brown', '-.')
    ppo = plot_from_species(path_local + "portfoliov0_all/ppo_" +'**', fig, 111, 'brown', '-.')
    a2c = plot_from_species(path_local + "portfoliov0_all/a2c_" +'**', fig, 111, 'brown', '-.')
    sac = plot_from_species(path_local + "portfoliov0_all/sac_" +'**', fig, 111, 'brown', '-.')
    
    # import ipdb; ipdb.set_trace()
    ax=fig.add_subplot(111)
    names = ["Rewards"]
    x = np.arange(len(names))  # the label locations
    width = 0.15  # the width of the bars
    rects1 = ax.bar(width+.25, np.mean(ppo, axis=0), width, label='PPO',yerr=np.std(ppo,axis=0))
    rects2 = ax.bar(width+.50, np.mean(a2c, axis=0), width, label='A2C',yerr=np.std(a2c,axis=0))
    rects2 = ax.bar(width+.75, np.mean(sac, axis=0), width, label='SAC',yerr=np.std(sac,axis=0))
    rects2 = ax.bar(width+.99, np.mean(random, axis=0), width, label='Random',yerr=np.std(random,axis=0))
    
    # ax.set_ylabel('Average accumulated distance')
    ax.set_title("Testing performance of different RL algorithms")
    ax.set_xticks(x+.22)
    ax.set_xticklabels(names)
    ax.legend(loc='best',  ncol=2)
    plt.savefig('bar_rewards.png')

if "assets" in drawlist:

    fig = figure()
    random = plot_from_file_assets(path_local + "portfoliov0_all/randomm_" +'**', fig, 111, 'brown', '-.')
    ppo = plot_from_species_assets(path_local + "portfoliov0_all/ppo_" +'**', fig, 111, 'brown', '-.')
    a2c = plot_from_species_assets(path_local + "portfoliov0_all/a2c_" +'**', fig, 111, 'brown', '-.')
    sac = plot_from_species_assets(path_local + "portfoliov0_all/sac_" +'**', fig, 111, 'brown', '-.')
    
    
    ax=fig.add_subplot(111)
    names = ["Assets"]
    x = np.arange(len(names))  # the label locations
    width = 0.15  # the width of the bars
    
    rects1 = ax.bar(width+.25, np.mean(ppo, axis=0), width, label='PPO',yerr=np.std(ppo,axis=0))
    rects2 = ax.bar(width+.50, np.mean(a2c, axis=0), width, label='A2C',yerr=np.std(a2c,axis=0))
    rects2 = ax.bar(width+.75, np.mean(sac, axis=0), width, label='SAC',yerr=np.std(sac,axis=0))
    rects2 = ax.bar(width+.99, np.mean(random, axis=0), width, label='Random',yerr=np.std(random,axis=0))

    
    ax.set_title("Testing performance of different RL algorithms")
    # ax.set_xticks(x+.22)
    ax.set_xticklabels(names)
    ax.legend(loc='best',  ncol=2)
    plt.savefig('bar_assets.png')