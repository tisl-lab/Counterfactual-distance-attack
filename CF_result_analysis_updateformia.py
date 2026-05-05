import pandas as pd
import argparse
import dill
import os
import ast
import sys

synth_Dss = ['synth_adult','synth_compass','synth_adult']



def parse_distance(distance):
    # If already a float, just return it
    if isinstance(distance, float) or isinstance(distance, int):
        return float(distance)
    # If it's a string, try to eval and extract the first element
    try:
        val = ast.literal_eval(distance)
        # If it's a list, return the first element
        if isinstance(val, list):
            return val[0]
        # If it's a float, return it
        if isinstance(val, float) or isinstance(val, int):
            return float(val)
    except Exception:
        pass
    # Fallback: try to convert directly
    try:
        return float(distance)
    except Exception:
        return None  # or np.nan
    
if __name__ == '__main__':

    #### load parameters
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='heloc', help='hospital,adult,informs,synth_adult')
    parser.add_argument('--rseed', type=int, default=2, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='NN', help='NN, RF, SVM, XgBoost')

    # get input      
    args = parser.parse_args()
    dataset_name = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model
    

    # datasets = ['compas','heloc','adult','acs_income']  #'hospital','adult',compas,'informs','synth_adult','synth_compass','synth_adult']
    datasets = ['heloc']  #'hospital','adult',compas,'informs','synth_adult','synth_compass','synth_adult']
    models = ['NN']  #['NN','RF']
    CF_method_list = ['NICE','dice_kdtree','dice_gradient','scfe'] # ['NICE','LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP','synth_dp_cf','ldp_ds_cf','zerocost_DP_CF']
    n_count_list = [0] #[0,3,5,10,20]
    one_neighbour_methods =  ['NICE','dice_kdtree','dice_gradient','scfe']#['LDP_CF','LDP_SRR','LDP_Noisy_max','NICE','synth_dp_cf','ldp_ds_cf','dice']
    multiple_neighbour_methods = ['inline_LDP','zerocost_DP_CF']
    # #### test data to check if it works
    # dataset_name = 'adult' #'synth_adult' #'hospital'#'adult'
    # RANDOM_SEED = 2
    
    # model = 'RF'
    # epsilon = 1
    NEIGHBOR_COUNT = 0
    # # instance_num = 12
    # CF_method = 'NICE'#'zerocost_DP_CF' #'DP_CF'#'LDP_CF'






    for dataset_name in datasets:
        for model in models:
            for RANDOM_SEED in range(0,5):
                resultsdir =    './dpnice/mia_explainer/cf_results/{}/'.format(dataset_name)
                processed_dir = './dpnice/mia_explainer/analysis/{}/'.format(dataset_name)
                processed_file = '{}{}_{}.pkl'.format(processed_dir,model,RANDOM_SEED)
    
                if not os.path.exists(processed_dir):
                    os.makedirs(processed_dir, exist_ok=True)

    
            #    for RANDOM_SEED in range (0 ,6):    #open resutls of every seed, process them and save them in your file
                all_datasets = []
                
                #for epsilon in (0.01,0.1,1,5,10):
                epsilon = 1    
                for CF_method in CF_method_list: #('LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP','NICE','zerocost_DP_CF','synth_dp_cf','ldp_ds_cf'):
                        resultfilepath = '{}{}/seed_{}.csv'.format(resultsdir, CF_method, RANDOM_SEED)
                        # processed_file = '{}{}/seed_{}.pkl'.format(processed_dir,CF_method,RANDOM_SEED)
                    # for NEIGHBOR_COUNT in n_count_list: # (0,3,5,10,20):
                        if(os.path.exists(resultfilepath)):
                            df = pd.read_csv(resultfilepath)
                            #First, generate array names, to know in file you should search for what
                            #generate array name
                            ## here we have a problem for k, so we should add 0 to k and epsilon to be able to process files correctly
                            if CF_method in CF_method_list:
                            #search in file for all related data
                                array_name = '{}_{}_{}'.format(CF_method,epsilon,NEIGHBOR_COUNT)
                                #add its name to the structure
                                # iterate over all df, read data, and extract information related to the array into that.
                                statistics_array =[]
                                
                                for index, row in df.iterrows():
                                    if (CF_method in one_neighbour_methods and NEIGHBOR_COUNT == 0): # or (CF_method in multiple_neighbour_methods and NEIGHBOR_COUNT != 0):
                                        if ((row['CF_method'] == CF_method and row['epsilon'] == epsilon  and row['k'] == NEIGHBOR_COUNT) or 
                                           (CF_method == 'scfe' and row['CF_method'] == CF_method and row['epsilon'] == epsilon  and row['k'] == 3)):
                                            #if the instance is a counterfactual, then add its data to the array
                                            if row['not_cf'] == False:
                                                # Access values from other columns for the rows with the special value
                                                distance = row['CF_distance']
                                                CF_distance = parse_distance(distance) #ast.literal_eval(distance)[0]
                                                reidentification_rate = row['cf__ri_count']
                                                changed_rate = row['changed_rate']
                                                same_inst_rate = row['same_inst_rate']
                                                kl_divergence = row['KL_divergence']
                                                min_dist = row['CF_min_dist']
                                                CF_min_dist = parse_distance(min_dist)# ast.literal_eval(min_dist)[0]
                                                min_k_dist = row['CF_min_k_dist']
                                                CF_min_k_dist = parse_distance(min_k_dist) #ast.literal_eval(min_k_dist)[0]
                                                rand_k_dist = row['CF_rand_k_dist']
                                                CF_rand_k_dist = parse_distance(rand_k_dist) #ast.literal_eval(rand_k_dist)[0]
                                                k = row['k']
                                                statistics_array.append([CF_distance,reidentification_rate,changed_rate,same_inst_rate,kl_divergence,k,CF_min_dist,CF_min_k_dist,CF_rand_k_dist])
                                                
                                            ### to make sure about the sizez, we can add zero/NA arrays for non_CF instances
                                        # else:
                                #### search in result file, if this array exsists, just append the data part, otherwise insert array    
                                if(statistics_array != []):
                                    onedatarow = {}
                                    onedatarow[array_name] = statistics_array
                                    all_datasets.append(onedatarow)
                                    print(f"file {resultfilepath} Processed")
                        else:
                            print(f"file {resultfilepath} does not exist")
                            continue
                if all_datasets != [] :
                    # m_dir = os.path.dirname(processed_file)
                    # if m_dir and not os.path.exists(m_dir):
                    #     os.makedirs(m_dir, exist_ok=True)
                    dill.dump(all_datasets, file=open(processed_file,"ab"))

