import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import types
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize 

# Model stuff
from sklearn import metrics
from sklearn.model_selection import train_test_split


from utils.analysis.evaluation import reidentify
from tools.save_results import generate_CF_record,save_CF_to_csv,add_to_CF_file
from utils.data.distance import heom_distance_new

from cf_dist_mia.utils import remove_colls
from cf_dist_mia.utils import load_data
from cf_dist_mia.utils import save_mia_results_to_csv

from cf_dist_mia.models import fit_model

from cf_dist_mia.counterfactuals import SCFE
from torch.utils.data import DataLoader
import torch.nn as nn

from cf_dist_mia.models import ann, Loader,predict_ann


import warnings
import argparse
import os
import sys
import pickle
import dill
import copy
import logging

from utils.data.load_data import Fetcher

from joblib import Parallel, delayed
import cloudpickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

INS_COUNT = 10000

# Module-level function for pickling compatibility
def process_instance_worker(args_dict):
    """Module-level worker function for parallel processing (picklable)"""
    j = args_dict['j']
    X_prime = args_dict['X_prime']
    preds_prime1 = args_dict['preds_prime1']
    recourse_model = args_dict['recourse_model']
    norm_factors = args_dict['norm_factors']
    scaler = args_dict['scaler']
    DS = args_dict['DS']
    X_train_shape_0 = args_dict['X_train_shape_0']
    X_test_shape_0 = args_dict['X_test_shape_0']
    cf_method = args_dict['cf_method']
    epsilon = args_dict['epsilon']
    
    thread_id = threading.current_thread().name
    start_time = time.time()
    # print(f"[{thread_id}] Starting instance {j}")
    
    """Process a single instance to generate counterfactuals"""
    pred_class = torch.tensor((preds_prime1[j] > 0.5) * 1)
    q_j = torch.tensor(X_prime[j,:]).reshape(1,-1).float()
    distance, cf = recourse_model.generate_counterfactuals_new(query_instance=q_j,
                                                    target_class=1-pred_class)
    
    record = None
    cf_original = None
    
    if distance is None:
        distance = np.inf
        no_cf = True
        record = generate_CF_record(cf_method, 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 
                                   not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                   same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
        cf_original = None
    else:
        no_cf = False
        cf_np = cf.detach().cpu().numpy().reshape(1, -1)
        cf_scaled = cf_np * norm_factors[X_train_shape_0 + X_test_shape_0 + j:X_train_shape_0 + X_test_shape_0 + j + 1]
        cf_original = scaler.inverse_transform(cf_scaled)
        
        for feat_idx in DS.dataset['categorical_features']:
            cf_original[0, feat_idx] = int(np.round(cf_original[0, feat_idx]))
        
        X_min = DS.X_train.min().values
        X_max = DS.X_train.max().values
        cf_original[0] = np.clip(cf_original[0], X_min, X_max)
        
        cf__ri_count = reidentify(cf_original, DS.X_train)
        scfe_df = pd.DataFrame(cf_original, columns=DS.dataset["feature_names"])
        orig_instance = DS.X_counterfactual.iloc[j:j+1,:]
        CF_distance = heom_distance_new(orig_instance.values[0], cf_original[0], 
                                       DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
        
        distances = [heom_distance_new(x.values.reshape(1, -1)[0], cf_original[0],
                                      DS.dataset['numerical_features'], DS.dataset['feature_ranges']) 
                    for _, x in DS.X_train.iterrows()]
        
        closest_idx = DS.X_train.index[np.argmin(distances)]
        closest_instance = DS.X_train.loc[closest_idx]
        plause_dist = heom_distance_new(closest_instance.values.reshape(-1,1)[0], cf_original[0],
                                       DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
        
        record = generate_CF_record(cf_method, scfe_df, CF_distance, plause_dist, 'NAN', 'NAN', 
                                   cf__ri_count, not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                   same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
    
    elapsed = time.time() - start_time
    # print(f"[{thread_id}] Completed instance {j} in {elapsed:.2f}s")
    return {'record': record, 'cf_original': cf_original, 'index': j}


def process_batch_worker(batch_args_dict):
    """Module-level worker function for batch processing (picklable)"""
    batch_indices = batch_args_dict['batch_indices']
    batch_args = batch_args_dict['args_dict_template']
    start_idx = batch_args_dict['start_idx']
    
    thread_id = threading.current_thread().name
    batch_start_time = time.time()
    # print(f"\n[{thread_id}] Starting batch {start_idx} with {len(batch_indices)} instances: {batch_indices}")
    
    batch_results = []
    for j in batch_indices:
        batch_args['j'] = j
        result = process_instance_worker(batch_args)
        batch_results.append(result)
    
    batch_elapsed = time.time() - batch_start_time
    print(f"[{thread_id}] Completed batch {start_idx} in {batch_elapsed:.2f}s")
    return start_idx, batch_results

###########
def load_dataset(dataset_name,rseed):
    
    DS = Fetcher(dataset_name) #adult,acs_income,heloc
    X = DS.dataset['X']
    Y = DS.dataset['y']
    
    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, Y, test_size=0.4,random_state=rseed)
    

    DS.X_train = X_train.copy()
    DS.y_train = y_train.copy()
    
    X_test,X_counterfactual,y_test,y_counterfactual=train_test_split(X_test_1, y_test_1, test_size=0.5,random_state=rseed)
    DS.X_test = X_test.copy()
    DS.y_test = y_test.copy()
    DS.X_counterfactual = X_counterfactual.copy()
    DS.y_counterfactual = y_counterfactual.copy()

    
    
    return DS


def fit_model(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array, params: dict, loss: str = 'log',DS=None,dice_data=None):
        dataset_train = Loader(X=X_train, 
                   y=Y_train[:].reshape(-1))
        trainloader = DataLoader(dataset_train, 
                                 batch_size=params['batch_size'], 
                                 shuffle=True)
        clf = ann(input_dim=X_train.shape[1], 
                  hidden_layers=params['hidden_layers'],
                  train_loader=trainloader,
                  epochs=params['epochs'])

        width = len(params['hidden_layers'])
        depth = params['hidden_layers'][0] 
        # print(X_train.shape)
        clf.fit(X=X_train,
                y=Y_train[:].reshape(-1))

        pred_train = ((clf(torch.from_numpy(X_train).float()) > 0.5) * 1).detach().numpy()
        pred_test = ((clf(torch.from_numpy(X_test).float()) > 0.5) *1 ).detach().numpy()
        clf.predict_proba = types.MethodType(predict_ann, clf)

        print("Accuracy on Train set:", accuracy_score(Y_train, pred_train))
        print("Accuracy on Test set:", accuracy_score(Y_test, pred_test))
        return clf

def  train_model(dataset,model_name, RANDOM_SEED,epsilon,cf_method):
        
    # 1- Load data
    d = None
    # dataname = params['dataset']
    # for dataname in params['datasets']:
    print(f'Computing results on: {dataset}')
    DS = load_dataset(dataset,0)

    X_train = DS.X_train.values.astype(float)
    Y_train = DS.y_train.values.astype(float)
    X_test = DS.X_test.values.astype(float)
    Y_test = DS.y_test.values.astype(float).astype(float)
    X_prime = DS.X_counterfactual.values.astype(float)
    Y_prime = DS.y_counterfactual.values.astype(float)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_prime = scaler.transform(X_prime)

    
    X_full = np.vstack([X_train, X_test, X_prime])
    norm_factors = np.linalg.norm(X_full, axis=1, keepdims=True)
    norm_factors = np.clip(norm_factors, 1e-12, None)
    
    X_full = X_full / norm_factors
    X_train = X_full[0:X_train.shape[0], :]
    X_test = X_full[X_train.shape[0]:X_train.shape[0]+X_test.shape[0], :]
    X_prime = X_full[X_train.shape[0]+X_test.shape[0]: , :]

    # logger.info(f"load dataset for scfe")
    # df, Y = load_data(dataset)
    #     # standardize data
    # scaler = StandardScaler()
    # X = df.values.astype(float)
    # scaler.fit(X)
    # X = scaler.transform(X)
    # # normalize data rowwise by ell 2 norm


    # #########
    # norm_factors = np.linalg.norm(X, axis=1, keepdims=True)
    # norm_factors = np.clip(norm_factors, 1e-12, None)
    # X = X / norm_factors
    # #########


    # X = normalize(X)
    # # split data
    # X_train, X_t, Y_train, Y_t = train_test_split(X, 
    #                                                 Y.astype(float), 
    #                                                 test_size=0.4,
    #                                                 random_state=RANDOM_SEED)

    # # Make sure 1/3 is hold out data
    # X_test, X_prime, Y_test, Y_prime = train_test_split(X_t, 
    #                                                     Y_t,
    #                                                     test_size=0.5,
    #                                                     random_state=RANDOM_SEED)

    ######################
    ###    FIT MODEL   ###
    ######################
    n_features = X_train.shape[1]
    params['categorical_features'] = []
    params['numerical_features'] = list(range(n_features))
    clf = fit_model(X_train, X_test, Y_train, Y_test, params)

    preds_train1 = clf(torch.from_numpy(X_train).float()).detach().numpy()
    preds_test1 = clf(torch.from_numpy(X_test).float()).detach().numpy()
    preds_prime1 = clf(torch.from_numpy(X_prime).float()).detach().numpy()
# choose recourse model
    recourse_model = SCFE(classifier=clf,
                            lr=params['lr_scfe'],
                            _lambda=0.0,
                            step=0.00,
                            max_iter=params['max_iter'],
                            norm=1,
                            target_threshold=0.5)
    

    
    indices = []
    instance_num = 0
    idx = 0
    while True:
    # for instance_num in range(1,101):
        # if(dataset_name not in synth_Dss):
        #     to_explain = DS.X_counterfactual[idx:idx+1 :] 
        # else:  #if we have used synthetic data to train the model, yet we need to explain real instances
        to_explain = DS.X_counterfactual[idx:idx+1 :] 
        # to_explain = DS.X_counterfactual[idx:idx+1 :] 
        if to_explain.empty:
            break
        # if NICE_model.data.predict_fn(to_explain).argmax() == 0:  # if model prediction is not desired 
        ### we want to generate a semi balances attack dataset, so we do not need for our instances to be in one class, we prefer diversity
        indices.append(idx)
        instance_num +=1
        if instance_num == INS_COUNT or idx> DS.X_counterfactual.shape[0] - 1: 
            if idx> DS.X_counterfactual.shape[0]:
                print("Not enough instances to generate CFs, Only {} instances are available".format(DS.X_counterfactual.shape[0]))
            break
        idx +=1


    resultsdir =    './dpnice/mia_explainer/cf_results/{}/{}/'.format(dataset,dataset)       
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    CF_file_dir = './dpnice/mia_explainer/cf_files/{}/{}/'.format(dataset,cf_method)
    if not os.path.exists(CF_file_dir):
        os.makedirs(CF_file_dir)
    CF_filename =  '{}seed_{}.csv'.format(CF_file_dir,RANDOM_SEED)
    resultfilename = '{}seed_{}.csv'.format(resultsdir, RANDOM_SEED)
    
   
    def process_instance(j, X_prime, preds_prime1, recourse_model, norm_factors, scaler, DS, X_train, X_test, cf_method, epsilon, params):
        """Process a single instance to generate counterfactuals"""
        pred_class = torch.tensor((preds_prime1[j] > 0.5) * 1)
        q_j = torch.tensor(X_prime[j,:]).reshape(1,-1).float()
        distance, cf = recourse_model.generate_counterfactuals_new(query_instance=q_j,
                                                        target_class=1-pred_class)
        
        record = None
        cf_original = None
        
        if distance is None:
            distance = np.inf
            no_cf = True
            record = generate_CF_record(cf_method, 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 
                                       not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                       same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
            cf_original = None
        else:
            no_cf = False
            cf_np = cf.detach().cpu().numpy().reshape(1, -1)
            cf_scaled = cf_np * norm_factors[X_train.shape[0] + X_test.shape[0] + j:X_train.shape[0] + X_test.shape[0] + j + 1]
            cf_original = scaler.inverse_transform(cf_scaled)
            
            for feat_idx in DS.dataset['categorical_features']:
                cf_original[0, feat_idx] = int(np.round(cf_original[0, feat_idx]))
            
            X_min = DS.X_train.min().values
            X_max = DS.X_train.max().values
            cf_original[0] = np.clip(cf_original[0], X_min, X_max)
            
            cf__ri_count = reidentify(cf_original, DS.X_train)
            scfe_df = pd.DataFrame(cf_original, columns=DS.dataset["feature_names"])
            orig_instance = DS.X_counterfactual.iloc[j:j+1,:]
            CF_distance = heom_distance_new(orig_instance.values[0], cf_original[0], 
                                           DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
            
            distances = [heom_distance_new(x.values.reshape(1, -1)[0], cf_original[0],
                                          DS.dataset['numerical_features'], DS.dataset['feature_ranges']) 
                        for _, x in DS.X_train.iterrows()]
            
            closest_idx = DS.X_train.index[np.argmin(distances)]
            closest_instance = DS.X_train.loc[closest_idx]
            plause_dist = heom_distance_new(closest_instance.values.reshape(-1,1)[0], cf_original[0],
                                           DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
            
            record = generate_CF_record(cf_method, scfe_df, CF_distance, plause_dist, 'NAN', 'NAN', 
                                       cf__ri_count, not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                       same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
        
        return {'record': record, 'cf_original': cf_original, 'index': j}
    
    # Batch parallel processing with fallback
    batch_size = params.get("batch_size", 20) if params else 20
    n = len(indices)
    batch_starts = list(range(0, n, batch_size))
    
    n_jobs = int(params.get("n_jobs", -1)) if params else -1
    all_results = []
    
    # Get actual number of jobs that will be used
    import multiprocessing
    max_cores = multiprocessing.cpu_count()
    actual_n_jobs = max_cores if n_jobs == -1 else (max_cores + n_jobs + 1 if n_jobs < 0 else n_jobs)
    actual_n_jobs = min(actual_n_jobs, len(batch_starts))  # Don't create more jobs than batches
    
    print(f"\n{'='*80}")
    print(f"PARALLEL PROCESSING CONFIGURATION:")
    print(f"  Total instances to process: {n}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(batch_starts)}")
    print(f"  Available CPU cores: {max_cores}")
    print(f"  Threads to use: {actual_n_jobs}")
    print(f"  Backend: ThreadPoolExecutor (no pickling required)")
    print(f"{'='*80}\n")
    
    parallel_start_time = time.time()
    try:
        # Use ThreadPoolExecutor for true threading without pickling
        with ThreadPoolExecutor(max_workers=actual_n_jobs) as executor:
            # Submit batch jobs
            future_to_batch_start = {}
            for start_idx in batch_starts:
                end_idx = min(start_idx + batch_size, n)
                batch_indices = indices[start_idx:end_idx]
                
                # Create closure for this batch
                def process_batch(batch_indices=batch_indices, start_idx=start_idx):
                    thread_id = threading.current_thread().name
                    batch_start_time = time.time()
                    print(f"\n[{thread_id}] Starting batch {start_idx} with {len(batch_indices)} instances: {batch_indices}")
                    
                    batch_results = []
                    for j in batch_indices:
                        start_time = time.time()
                        # print(f"[{thread_id}] Starting instance {j}")
                        
                        # Process instance
                        pred_class = torch.tensor((preds_prime1[j] > 0.5) * 1)
                        q_j = torch.tensor(X_prime[j,:]).reshape(1,-1).float()
                        distance, cf = recourse_model.generate_counterfactuals_new(query_instance=q_j,
                                                                        target_class=1-pred_class)
                        
                        record = None
                        cf_original = None
                        
                        if distance is None:
                            distance = np.inf
                            no_cf = True
                            record = generate_CF_record(cf_method, 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 
                                                       not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                                       same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
                            cf_original = None
                        else:
                            no_cf = False
                            cf_np = cf.detach().cpu().numpy().reshape(1, -1)
                            cf_scaled = cf_np * norm_factors[X_train.shape[0] + X_test.shape[0] + j:X_train.shape[0] + X_test.shape[0] + j + 1]
                            cf_original = scaler.inverse_transform(cf_scaled)
                            
                            for feat_idx in DS.dataset['categorical_features']:
                                cf_original[0, feat_idx] = int(np.round(cf_original[0, feat_idx]))
                            
                            X_min = DS.X_train.min().values
                            X_max = DS.X_train.max().values
                            cf_original[0] = np.clip(cf_original[0], X_min, X_max)
                            
                            cf__ri_count = reidentify(cf_original, DS.X_train)
                            scfe_df = pd.DataFrame(cf_original, columns=DS.dataset["feature_names"])
                            orig_instance = DS.X_counterfactual.iloc[j:j+1,:]
                            CF_distance = heom_distance_new(orig_instance.values[0], cf_original[0], 
                                                           DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
                            
                            distances = [heom_distance_new(x.values.reshape(1, -1)[0], cf_original[0],
                                                          DS.dataset['numerical_features'], DS.dataset['feature_ranges']) 
                                        for _, x in DS.X_train.iterrows()]
                            
                            closest_idx = DS.X_train.index[np.argmin(distances)]
                            closest_instance = DS.X_train.loc[closest_idx]
                            plause_dist = heom_distance_new(closest_instance.values.reshape(-1,1)[0], cf_original[0],
                                                           DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
                            
                            record = generate_CF_record(cf_method, scfe_df, CF_distance, plause_dist, 'NAN', 'NAN', 
                                                       cf__ri_count, not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                                       same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
                        
                        elapsed = time.time() - start_time
                        # print(f"[{thread_id}] Completed instance {j} in {elapsed:.2f}s")
                        batch_results.append({'record': record, 'cf_original': cf_original, 'index': j})
                    
                    batch_elapsed = time.time() - batch_start_time
                    print(f"[{thread_id}] Completed batch {start_idx} in {batch_elapsed:.2f}s")
                    return start_idx, batch_results
                
                future = executor.submit(process_batch)
                future_to_batch_start[future] = start_idx
            
            # Collect results as they complete
            print(f"Starting parallel execution with {actual_n_jobs} threads...\n")
            for future in as_completed(future_to_batch_start):
                start_idx, results = future.result()
                all_results.extend(results)
                if (start_idx % 100) == 0:
                    print(f"Finding counterfactual for {cf_method} at iteration: {start_idx}")
                    
    except Exception as exc:
        print(f"\nParallel execution failed: {exc}. Falling back to serial execution.")
        # Fallback to serial execution on any parallel error
        for start_idx in batch_starts:
            end_idx = min(start_idx + batch_size, n)
            batch_indices = indices[start_idx:end_idx]
            thread_id = "MainThread-Serial"
            batch_start_time = time.time()
            print(f"\n[{thread_id}] Starting batch {start_idx} with {len(batch_indices)} instances: {batch_indices}")
            
            for j in batch_indices:
                start_time = time.time()
                # print(f"[{thread_id}] Starting instance {j}")
                
                # Process instance
                pred_class = torch.tensor((preds_prime1[j] > 0.5) * 1)
                q_j = torch.tensor(X_prime[j,:]).reshape(1,-1).float()
                distance, cf = recourse_model.generate_counterfactuals_new(query_instance=q_j,
                                                                target_class=1-pred_class)
                
                record = None
                cf_original = None
                
                if distance is None:
                    distance = np.inf
                    no_cf = True
                    record = generate_CF_record(cf_method, 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 'NAN', 
                                               not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                               same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
                    cf_original = None
                else:
                    no_cf = False
                    cf_np = cf.detach().cpu().numpy().reshape(1, -1)
                    cf_scaled = cf_np * norm_factors[X_train.shape[0] + X_test.shape[0] + j:X_train.shape[0] + X_test.shape[0] + j + 1]
                    cf_original = scaler.inverse_transform(cf_scaled)
                    
                    for feat_idx in DS.dataset['categorical_features']:
                        cf_original[0, feat_idx] = int(np.round(cf_original[0, feat_idx]))
                    
                    X_min = DS.X_train.min().values
                    X_max = DS.X_train.max().values
                    cf_original[0] = np.clip(cf_original[0], X_min, X_max)
                    
                    cf__ri_count = reidentify(cf_original, DS.X_train)
                    scfe_df = pd.DataFrame(cf_original, columns=DS.dataset["feature_names"])
                    orig_instance = DS.X_counterfactual.iloc[j:j+1,:]
                    elapsed = time.time() - start_time
                    CF_distance = heom_distance_new(orig_instance.values[0], cf_original[0], 
                                                   DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
                    
                    distances = [heom_distance_new(x.values.reshape(1, -1)[0], cf_original[0],
                                                  DS.dataset['numerical_features'], DS.dataset['feature_ranges']) 
                                for _, x in DS.X_train.iterrows()]
                    
                    closest_idx = DS.X_train.index[np.argmin(distances)]
                    closest_instance = DS.X_train.loc[closest_idx]
                    plause_dist = heom_distance_new(closest_instance.values.reshape(-1,1)[0], cf_original[0],
                                                   DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
                    
                    record = generate_CF_record(cf_method, scfe_df, CF_distance, plause_dist, 'NAN', 'NAN', 
                                               cf__ri_count, not_cf=no_cf, epsilon=epsilon, k=3, changed_rate='NA', 
                                               same_inst_rate='NA', SameAsNice='NA', index=j, iter=j)
                
                elapsed = time.time() - start_time
                # print(f"[{thread_id}] Completed instance {j} in {elapsed:.2f}s (serial fallback)")
                all_results.append({'record': record, 'cf_original': cf_original, 'index': j})
            
            batch_elapsed = time.time() - batch_start_time
            print(f"[{thread_id}] Completed batch {start_idx} in {batch_elapsed:.2f}s (serial fallback)")
            
            if (start_idx % 100) == 0:
                print(f"Finding counterfactual for {cf_method} at iteration: {start_idx} (serial fallback)")
    
    parallel_elapsed = time.time() - parallel_start_time
    print(f"\n{'='*80}")
    print(f"PARALLEL PROCESSING COMPLETED in {parallel_elapsed:.2f}s")
    print(f"{'='*80}\n")
    
    # Save results
    distances = np.zeros(idx + 1)
    for result in all_results:
        if result['record'] is not None:
            save_CF_to_csv(record=result['record'], csv_file_path=resultfilename)
            if result['cf_original'] is not None:
                add_to_CF_file(CF_filename, result['cf_original'], columns_list=DS.dataset['feature_names'])

    # cfs_arr = np.vstack([cf.detach().cpu().numpy().reshape(1, -1) for cf in cfs])
    # cfs_scaled = cfs_arr * norm_factors[:cfs_arr.shape[0]]  # undo row-wise L2 normalization
    # cfs_original = scaler.inverse_transform(cfs_scaled) 
    
    return distances
    # return recourse_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="acs_income",
        help="Dataset to use: adult, acs_income, heloc, compas",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ann",
        help="Model name to use: ann, rf, xgb, lr",
    )
    parser.add_argument(
        "--rseed",
        type=int,
        default=4,
        help="Random seed for data splitting and model training",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1,
        help="Epsilon value for SCFE method",
    )
    parser.add_argument(
        "--cf_method",
        type=str,
        default="scfe",
        help="Counterfactual generation method to use",
    )

    args = parser.parse_args()

    # Define model training parameters
    params = {
        'batch_size': 64,
        'hidden_layers': [32, 32],
        'epochs': 50,
    }
    params = {
        'experiment': 'realworld',                                      # options: {'synthetic', 'realworld'}
        'dataset': args.dataset,  #['mnist', 'heloc', 'default', 'housing'],           # options: datasets: {'churn', 'mnist', 'heloc', 'default', 'housing'}
        'method': [args.cf_method],#CF_method,   'scfe',  dice_gradient,NICE                                       # current options for linear models: {'scfe','cchvae'}; for nonlinear models: {'scfe'}
        'fit_sgd': False,                                               # boolean: whether linear model should be fitted with sgd
        'linear': False, # True,                                                 # boolean: whether linear or nonlinear model should be fitted
        'hidden_layers': [100] * 1,                                     # list with ints: size of hidden layers (for nonlinear model)
        'epochs': 50,                                                  # int: number epochs (for nonlinear model)
        'batch_size': 32,                                               # int: batch size (for nonlinear model)
        'epsd': 1e-5,                                                   # pos real: constant to ensure stable loss evaluations
        'weighting': 'global',                                          # options: {'global', 'local'}: whether global variance or 1/n_shadow_models variance
        'n_shadow_models': 10,                                                                                     # int: number of shadow models
        'frac': 0.75,                                                   # frac: fraction of samples used for resampling scheme within shadow model train pipeline
        'penalty': 'none',                                             # options: {'none', 'l1', 'l2'}: reguralizer for underlying model
        'C': 1000,                                                      # pos real: regularization strength: higher leads less regularized models
        'ensemble': False,                                               # boolean: whether to fit an underlying ensemble model
        'n_ensemble': 1, # 250,                                              # int: number of ensemble models
        # 'frac_ensemble': shadow_frac, #0.75,                                          # pos real in [0,1]: fraction of samples used for resampling to train ensemble model
        'disjoint': True,                                               # boolean: Whether disjoint data sets should be used for training private ensemble
        'n_splits': 3,                                                  # int: Number of disjoint ensembles
        'lr_scfe': 0.005,                                               # pos. real in [0,1]:Learning rate for SCFE gradient descent
        'max_iter': 1000,                                               # int: max number of iterations for SCFE gradient descent
        'quantile': 0.97,      # Number of splits for disjoint datasets
        'n_estimators': 100 ,
        'random_state': 42 ,
        # 'attack_set_size': attack_size,    ## the size of subsample for which attack is performed
        'max_depth': 10  # Number of splits for disjoint datasets
    }
    train_model(args.dataset, args.model_name, args.rseed, args.epsilon, args.cf_method)

    print("All cfs generated. for dataset {} seed {}".format(args.dataset,args.rseed))



    