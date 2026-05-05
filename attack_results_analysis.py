import pandas as pd
import argparse
import dill
import os
# import ast
import sys
import numpy as np
# import matplotlib.pyplot as plt
# from utils.analysis.draw_plots import draw_CDF
from tools.save_results import save_CF_to_csv


if __name__ == '__main__':

    #### load parameters
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='heloc', help='hospital,adult,informs,synth_adult')
    # parser.add_argument('--rseed', type=int, default=2, help='random seed: choose between 0 - 5')
    # parser.add_argument('--model', type=str, default='RF', help='NN, RF, SVM, XgBoost')

    # get input      
    args = parser.parse_args()
    dataset_name = args.dataset
    # rseed = args.rseed
    

    # CF_method_list = ['NICE','dice_kdtree','dice_gradient'] # ['NICE','LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP','synth_dp_cf','ldp_ds_cf','zerocost_DP_CF']
    # for dataset_name in datasets:
    # for model in models:
    attackresultsdir =  'dpnice/experiments/attack_results/{}/'.format(dataset_name)
    analysed_dir = 'dpnice/experiments/attack_results/{}/'.format(dataset_name)
    analysed_file = '{}averages_results'.format(analysed_dir)
    if not os.path.exists(analysed_dir):
            os.makedirs(analysed_dir, exist_ok=True)
    all_datasets = []
    for rseed in range(0,5):        
        attack_result_filepath = '{}{}/cf_dist_attack_results.csv'.format(attackresultsdir,rseed)
    #    for RANDOM_SEED in range (0 ,6):    #open resutls of every seed, process them and save them in your file
  
        # resultfilepath = '{}{}_{}.csv'.format(resultsdir, model, RANDOM_SEED)
        try:
            df = pd.read_csv(attack_result_filepath)    
                
            all_datasets.append(df.assign(rseed=rseed))
            
        except Exception:
            print("File not found: {}".format(attack_result_filepath))
            continue 
    
    
   
    final_df = pd.concat(all_datasets, ignore_index=True)

    
    if 'seed' in final_df.columns:
        final_df = final_df.drop(columns=['seed'])
    avg_df = final_df.groupby(
        ['dataset', 'method', 'attack_method','attack_set_size'], as_index=False
    ).mean(numeric_only=True)
    avg_df.to_csv(f"{analysed_file}.csv", index=False)
