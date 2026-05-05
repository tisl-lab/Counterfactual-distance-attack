import pandas as pd
import numpy as np
import dill
import argparse
import csv
import os
import ast
import re
import sys
import types

from utils.analysis.evaluation import reidentify

def _ensure_pandas_index_numeric_shim():
    """Provide a compatibility shim for pickles referencing pandas.core.indexes.numeric."""
    if 'pandas.core.indexes.numeric' in sys.modules:
        return
    shim = types.ModuleType('pandas.core.indexes.numeric')
    # Best-effort aliases across pandas versions
    try:
        from pandas import Int64Index  # pandas < 2
    except Exception:
        from pandas import Index as Int64Index
    try:
        from pandas import UInt64Index
    except Exception:
        UInt64Index = Int64Index
    try:
        from pandas import Float64Index
    except Exception:
        Float64Index = Int64Index
    from pandas import Index, RangeIndex
    shim.Int64Index = Int64Index
    shim.UInt64Index = UInt64Index
    shim.Float64Index = Float64Index
    shim.Index = Index
    shim.RangeIndex = RangeIndex
    sys.modules['pandas.core.indexes.numeric'] = shim

def load_ds_file(dataset, seed,epsilon = 1):
    ds_file_dir = './dpnice/mia_dataset_loaded/{}/'.format(dataset)
    ds_filename = '{}seed_{}.pkl'.format(ds_file_dir, seed)
    with open(ds_filename, 'rb') as file:
        _ensure_pandas_index_numeric_shim()
        try:
            ds = dill.load(file)
        except ModuleNotFoundError as e:
            # Retry after shim for older pandas pickles
            if "pandas.core.indexes.numeric" in str(e):
                file.seek(0)
                _ensure_pandas_index_numeric_shim()
                ds = dill.load(file)
            else:
                raise
    return ds

# to ensure both members and non-members are included in the attack dataset
def merge_and_shuffle_cf_files(model, dataset, seed, epsilon, cf_method, n_count):
    # Load the regular and basic CF files
    regular_cf_df = load_cf_file(model, dataset, seed, epsilon, cf_method, n_count, file_type='regular')
    basic_cf_df = load_cf_file(model, dataset, seed, epsilon, cf_method, n_count, file_type='basic')
    
    # Concatenate the two DataFrames
    merged_df = pd.concat([regular_cf_df, basic_cf_df], axis=0).reset_index(drop=True)
    
    # Shuffle the merged DataFrame
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    
    return shuffled_df

def load_cf_files(model, dataset, seed, epsilon, cf_method, n_count):
    # Load the regular and basic CF files
    regular_cf_df = load_cf_file(model, dataset, seed, epsilon, cf_method, n_count, file_type='regular')
    # basic_cf_df = load_cf_file(model, dataset, seed, epsilon, cf_method, n_count, file_type='basic')
    
    # Concatenate the two DataFrames
    # merged_df = pd.concat([regular_cf_df, basic_cf_df], axis=0).reset_index(drop=True)
    
    # Shuffle the merged DataFrame
    # shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    
    return regular_cf_df

def extract_members_non_members(cf_df):
    members = cf_df[cf_df['is_member'] == 1].drop(columns=['is_member'])
    non_members = cf_df[cf_df['is_member'] == 0].drop(columns=['is_member'])
    return members, non_members

def sample_from_dataset_set(cf_method, dataset, seed, count, membership = False,aux_data = False):
    ds = load_ds_file(dataset, seed)
    if aux_data:
        n_seed = np.random.seed(abs(500 - ( seed * 3 )))
        df = pd.DataFrame(ds.X_test, columns=ds.dataset['feature_names'])
        samples = df.sample(n=count, random_state=n_seed).reset_index(drop=True)
        return samples
    else:
        np.random.seed(seed)
    # ds = load_ds_file(model, dataset, seed)
    
        if membership:
            # Convert to DataFrame and extract instances from the training dataset
            df = pd.DataFrame(ds.X_train, columns=ds.dataset['feature_names'])
            samples = df.sample(n=count, random_state=seed).reset_index(drop=True)
            num_feat = ds.dataset['numerical_features']
            cat_feat = ds.dataset['categorical_features']
            return samples,num_feat,cat_feat
        else:
            # Extract instances from the test dataset
            nm_df = pd.DataFrame(ds.X_test, columns=ds.dataset['feature_names'])
            nm_df = nm_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the CF dataset
            
            samples = pd.DataFrame(columns=ds.dataset['feature_names'])
            for row in nm_df.values:
                if len(samples) >= count * 2:
                    break
                # Validate reidentification rate
                reidentification_rate = reidentify(row.reshape(1,-1), ds.X_train)
                if reidentification_rate == 0:  # Ensure no similar instance in the training dataset
                    samples = pd.concat([samples, pd.DataFrame([row], columns=ds.dataset['feature_names'])], ignore_index=True)
            return samples
    

def load_cf_file(dataset,cf_method, seed):
    cf_file_dir = './dpnice/mia_explainer/cf_files/{}/{}/'.format(dataset,cf_method)
    cf_filename =  '{}seed_{}.csv'.format(cf_file_dir,seed)
    return pd.read_csv(cf_filename)



    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MIA attack')
    parser.add_argument('--model', type=str, default='NN', help='Model type (e.g., RF, NN, XgBoost)')
    parser.add_argument('--dataset', type=str, default='acs_income', help='Dataset name (e.g., adult, hospital)')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed')
    parser.add_argument('--epsilon', type=float, default=1, help='Privacy parameter epsilon')
    parser.add_argument('--CF_method', type=str, default='scfe', help='dice_kdtree,dice_genetic,dice_gradient,NICE,NICE_real,inline_LDP,Counterfactual method')
    parser.add_argument('--n_count', type=int, default=3, help='Number of counterfactuals')
    parser.add_argument('--pct_real_id', type=float, default=1, help='Percentage of real data to recover with MIA')
    parser.add_argument('--attack_type', type=str, default='gan_leak_attack', help='domias_attack,monte_carlo_attack,gan_leak_attack')
    parser.add_argument('--attack_set_size', type=str, default='1000', help='1-biggest cf size, attack set size')

    args = parser.parse_args()
    

    dataset = args.dataset
    seed = args.rseed
    model = args.model
    epsilon = args.epsilon
    cf_method = args.CF_method
    n_count = args.n_count
    pct_real_id = args.pct_real_id
    attack_type = args.attack_type
    attack_set_size = int(args.attack_set_size)
    ds_path = './dpnice/mia_dataset_loaded/{}/'.format(dataset)
    file_path = './dpnice/mia_attack_ds/{}/'.format(dataset)    
    os.makedirs(file_path, exist_ok=True)
    cf_method_dir = os.path.join(file_path,str(seed))
    os.makedirs(cf_method_dir, exist_ok=True)
    #dataset is loaded to sample members from training dataset and non-members from counterfactual dataset
    ds = load_ds_file(dataset, seed)
    
    # Extract members and non-members from the attack_df to be able to evaluate the attack

    synth_ds = load_cf_file(dataset, cf_method,seed)
    synth_filename = '{}/{}/{}/{}/synth_1x.csv'.format(file_path,model,seed,cf_method)
    os.makedirs(os.path.dirname(synth_filename), exist_ok=True)
    synth_ds.to_csv('{}'.format(synth_filename), index=False)

    # members,numerical_features,categorical_features= sample_from_dataset_set(cf_method, dataset, seed,count = attack_set_size,membership = True)
    # nonmembers = sample_from_dataset_set(model, dataset, seed,count = attack_set_size,membership = False)
    # # These saved files can be used for all attack implementations    
    # ### members and non-members are sampled from the training dataset and counterfactual dataset respectively
    # members.to_csv('{}/{}/mem_set.csv'.format(file_path,seed), index=False)
    # nonmembers.to_csv('{}/{}/holdout_set.csv'.format(file_path,seed), index=False)
    # pd.Series(numerical_features, name='numerical_features').to_csv(
    # '{}/numerical_features.csv'.format(file_path), index=False)
    # pd.Series(categorical_features, name='categorical_features').to_csv(
    # '{}/categorical_features.csv'.format(file_path), index=False)
    # ## we may need to merge these to for attack dataset

    # aux_size = int(attack_set_size) # 10% of the attack set size
    #     #### update to sample from test/cf set instead of training set
    # df_aux = sample_from_dataset_set(model, dataset, seed,count = aux_size, aux_data=True)
    # df_aux.to_csv('{}/{}/aux_data.csv'.format(file_path,seed), index=False)