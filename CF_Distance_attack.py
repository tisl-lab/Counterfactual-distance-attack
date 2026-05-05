import warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize 

# Model stuff
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

from cf_dist_mia.utils import pipeline

from cf_dist_mia.utils import remove_colls
from cf_dist_mia.utils import load_data
from cf_dist_mia.utils import save_mia_results_to_csv
from cf_dist_mia.synth_data import gen_lin_data
from cf_dist_mia.models import fit_model


from utils.analysis.evaluation import AttackEvaluator
from utils.analysis.classifier import classifier

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore') 
import argparse
import os
import sys
import pickle
import dill
import copy
import logging

from utils.data.load_data import Fetcher

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cf_distance_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


'''
Parameters to control:
-> data set type
-> model type {linear, nonlinear}
  -> if linear:
    -> if: ensemble
        -> disjoint data split
        -> randomly subsampled data (not disjoint)
    -> else: vanilla model
  -> else: nonlinear vanilla model
-> recourse methods
    -> SCFE
    -> GS
    -> CCHVAE (called 'latent' here)
<<<NOTE: THE SCRIPT ITERATES OVER ALL OF THESE>>>
-> attack methods
    -> CFD attack {our suggestion}
    -> CFD LRT {our suggestion}
    -> vanilla loss attack
    -> LRT loss attack {SOTA}
'''
#default parameters, it gets updated based on input arguments
params = {
    'experiment': 'realworld',                                      # options: {'synthetic', 'realworld'}
    'datasets': {'acs_income'},  #[acs_income,'mnist', 'heloc', 'default', 'housing'],           # options: datasets: {'churn', 'mnist', 'heloc', 'default', 'housing'}
    'methods': {'NICE'},                                            # current options for linear models: {'scfe','cchvae'}; for nonlinear models: {'scfe'}
    'fit_sgd': False,                                               # boolean: whether linear model should be fitted with sgd
    'linear': False, # True,                                                 # boolean: whether linear or nonlinear model should be fitted
    'hidden_layers': [100] * 1,                                     # list with ints: size of hidden layers (for nonlinear model)
    'epochs': 750,                                                  # int: number epochs (for nonlinear model)
    'batch_size': 32,                                               # int: batch size (for nonlinear model)
    'epsd': 1e-5,                                                   # pos real: constant to ensure stable loss evaluations
    'weighting': 'global',                                          # options: {'global', 'local'}: whether global variance or 1/n_shadow_models variance
    'n_shadow_models': 2,                                           # int: number of shadow models
    'frac': 0.75,                                                   # frac: fraction of samples used for resampling scheme within shadow model train pipeline
    'penalty': 'l2',                                              # options: {'none', 'l1', 'l2'}: reguralizer for underlying model
    'C': 1000,                                                      # pos real: regularization strength: higher leads less regularized models
    'ensemble': False,                                               # boolean: whether to fit an underlying ensemble model
    'n_ensemble': 250,                                              # int: number of ensemble models
    'frac_ensemble': 1,                                          # pos real in [0,1]: fraction of samples used for resampling to train ensemble model
    'disjoint': True,                                               # boolean: Whether disjoint data sets should be used for training private ensemble
    'n_splits': 3,                                                  # int: Number of disjoint ensembles
    'lr_scfe': 0.005,                                               # pos. real in [0,1]:Learning rate for SCFE gradient descent
    'max_iter': 1000,                                               # int: max number of iterations for SCFE gradient descent
    'quantile': 0.97                                                # pos real in [0,1]: quantile to be used for quantile-based thresholds
}

# Parameters to control the synthetic data generating process described in the README.md
synth_gauss = {    
    'n': 240,                              # n_train -> 5000
    'd': [50, 250, 1500, 1800, 3000],      # data dimensions
    'sigma2': 1,                           # data variance
    'frac_important_features': 0.1,        # fraction of relevant features
    'sigma2_eps': 0.01,                    # label noise
    'corr': 0.0,                           # correlation between features
}

###########
def load_dataset(dataset_name):
    
    DS = Fetcher(dataset_name) #adult,acs_income,heloc
    X = DS.dataset['X']
    Y = DS.dataset['y']
    
    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, Y, test_size=0.4,random_state=args.rseed)
    

    
    DS.X_train = X_train.copy()
    DS.y_train = y_train.copy()
    
    X_test,X_counterfactual,y_test,y_counterfactual=train_test_split(X_test_1, y_test_1, test_size=0.5,random_state=args.rseed)
    DS.X_test = X_test.copy()
    DS.y_test = y_test.copy()
    DS.X_counterfactual = X_counterfactual.copy()
    DS.y_counterfactual = y_counterfactual.copy()

    
    
    return DS
###########
def subsample_data(X, y, target_size=None, target_fraction=None, random_state=42, stratify=True):
    """
    Subsample data by target size or fraction.
    
    Args:
        X: Feature array
        y: Label array
        target_size: Number of samples to keep (absolute). If None, uses target_fraction
        target_fraction: Fraction of samples to keep (0-1). Used only if target_size is None
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified sampling (maintains class distribution)
    
    Returns:
        X_sampled, y_sampled: Subsampled data
    
    Raises:
        ValueError: If both target_size and target_fraction are None
    """
    if X is None or y is None:
        return X, y
    
    actual_size = X.shape[0]
    
    # Determine sampling size
    if target_size is not None:
        sample_size = min(target_size, actual_size)
        if actual_size < target_size:
            print(f"⚠️  Dataset size ({actual_size}) < target size ({target_size}). Using all {actual_size} samples.")
    elif target_fraction is not None:
        if not 0 < target_fraction <= 1:
            raise ValueError(f"target_fraction must be in (0, 1], got {target_fraction}")
        sample_size = max(1, int(actual_size * target_fraction))
    else:
        raise ValueError("Either target_size or target_fraction must be provided")
    
    # If sample size equals actual size, return all data
    if sample_size >= actual_size:
        print(f"✓ Using all {actual_size} samples (no subsampling needed)")
        return X, y
    
    # Perform sampling
    try:
        if stratify:
            X_sampled, _, y_sampled, _ = train_test_split(
                X, y,
                train_size=sample_size,
                random_state=random_state,
                stratify=y
            )
        else:
            X_sampled, _, y_sampled, _ = train_test_split(
                X, y,
                train_size=sample_size,
                random_state=random_state
            )
        print(f"✓ Sampled {sample_size} / {actual_size} samples ({100*sample_size/actual_size:.1f}%)")
        return X_sampled, y_sampled
        
    except Exception as e:
        print(f"⚠️  Stratified sampling failed: {e}. Attempting without stratification...")
        X_sampled, _, y_sampled, _ = train_test_split(
            X, y,
            train_size=sample_size,
            random_state=random_state
        )
        print(f"✓ Sampled {sample_size} / {actual_size} samples ({100*sample_size/actual_size:.1f}%)")
        return X_sampled, y_sampled
########### From here we shoold update for our experiments ###########
def experiment(params, target_model , target_dataset, synth_gauss):

    scores = {
        
        'dists_train_NICE': [],              # DISTANCES
        'dists_test_NICE': [],
        'dists_train_dice_gradient': [],
        'dists_test_dice_gradient': [],
        'runtime_dists_dice_gradient': 0,
        'dists_train_dice_kdtree': [],    
        'dists_test_dice_kdtree': [],
        'runtime_dists_dice_kdtree': 0,
        'dists_train_inline_dp': [],
        'dists_test_inline_dp': [],
        'runtime_dists_inline_dp': 0,
        'dists_train_scfe': [],              # DISTANCES
        'dists_test_scfe': [],
        'runtime_dists_scfe': 0,
        'dists_lrt_train_local_NICE': [],
        'dists_lrt_test_local_NICE': [],
        'dists_lrt_train_global_NICE': [],
        'dists_lrt_test_global_NICE': [],
        'runtime_dists_lrt_NICE_global': 0,
        'runtime_dists_lrt_NICE_local': 0,
        'dists_lrt_train_local_inline_dp': [],
        'dists_lrt_test_local_inline_dp': [],
        'dists_lrt_train_global_inline_dp': [],
        'dists_lrt_test_global_inline_dp': [],
        'runtime_dists_lrt_inline_dp_global': 0,
        'runtime_dists_lrt_inline_dp_local': 0,
        'dists_lrt_train_local_dice_gradient': [],
        'dists_lrt_test_local_dice_gradient': [],
        'dists_lrt_train_global_dice_gradient': [],
        'dists_lrt_test_global_dice_gradient': [],
        'runtime_dists_lrt_dice_gradient_global': 0,
        'runtime_dists_lrt_dice_gradient_local': 0,
        'dists_lrt_train_local_dice_kdtree': [],
        'dists_lrt_test_local_dice_kdtree': [],
        'dists_lrt_train_global_dice_kdtree': [],
        'dists_lrt_test_global_dice_kdtree': [],
        'runtime_dists_lrt_dice_kdtree_global': 0,
        'runtime_dists_lrt_dice_kdtree_local': 0,
        'dists_lrt_train_local_scfe': [],
        'dists_lrt_test_local_scfe': [],
        'dists_lrt_train_global_scfe': [],
        'dists_lrt_test_global_scfe': [],
        'runtime_dists_lrt_scfe_global': 0,
        'runtime_dists_lrt_scfe_local': 0,
        'dists_train_cchvae': [],
        'dists_test_cchvae': [],
        'runtime_dists_cchvae': 0,
        'dists_lrt_train_local_cchvae': [],
        'dists_lrt_test_local_cchvae': [],
        'dists_lrt_train_global_cchvae': [],
        'dists_lrt_test_global_cchvae': [],
        'runtime_dists_lrt_cchvae': 0,
        'dists_train_gs': [],
        'dists_test_gs': [],
        'runtime_dists_gs': 0,
        'dists_lrt_train_local_gs': [],
        'dists_lrt_test_local_gs': [],
        'dists_lrt_train_global_gs': [],
        'dists_lrt_test_global_gs': [],       
        'runtime_dists_lrt_gs': 0,
        'losses_train': [],                  # LOSSES
        'losses_test': [],
        'runtime_losses': 0,
        'stable_losses_train': [],
        'stable_losses_test': [],
        'runtime_stable_losses': 0,
        'losses_lrt_train_local': [],
        'losses_lrt_test_local': [],
        'losses_lrt_train_global': [],
        'losses_lrt_test_global': [],
        'runtime_losses_lrt': 0
    }


    #####################
    ###   LAOD DATA   ###
    #####################
    
    '''
    Below the data is not only loaded but also cleaned. 
    We remove columns that lead to multicollinearity problems in linear models.
    After the data is loaded and cleaned, we normalize and split the data into 
    three equal folds. The third fold is usually not necessary, but we used it for
    the quantile strategy (not in the paper). To be consistent across all experiments
    we then applied this splitting strategy across all datasets.
    '''
    ####  update it, what should we do? we have train, test, queries and counterfactuals. 
    ### no need to train original models and generate counterfactuals.
    ### only train shadow models and find their distribution.   
    ## our models are nonlinear, does it mean each query needs shadow model?

    # 1- Load data
    d = None
    dataname = params['dataset']
    # for dataname in params['datasets']:
    print(f'Computing results on: {dataname}')
    # load model, dataset, queries, counterfactuals
    # if params['method'] in ('NICE','dice_gradient','inline_DP','dice_kdtree'):
    if 'dice_gradient' in params['method']:
        logger.info(f"load dataset for dice methods")
        # with open(target_model, 'rb') as file:
        #     clf = dill.load(file)
        #load the dataset
        # with open(target_dataset, 'rb') as file:
        #     DS = dill.load(file)
        DS = load_dataset(dataset_name)
        
        X_train = DS.X_train
        X_test = DS.X_test
        Y_train = DS.y_train
        Y_test = DS.y_test
        X_prime = DS.X_counterfactual
        Y_prime = DS.y_counterfactual
        # if params['method'] in ('dice_gradient','dice_kdtree'):

        params['numerical_features'] = DS.dataset['numerical_features']
        params['categorical_features'] = DS.dataset['categorical_features']
        params['feature_names'] = DS.dataset['feature_names']
        clf,d = fit_model(X_train, X_test, Y_train, Y_test, params,DS=DS)
    elif 'dice_kdtree' in params['method']:
        logger.info(f"load dataset for nice")
        DS = load_dataset(dataset_name)
        X_train = DS.X_train
        X_test = DS.X_test
        Y_train = DS.y_train
        Y_test = DS.y_test
        X_prime = DS.X_counterfactual
        Y_prime = DS.y_counterfactual
        # if params['method'] in ('dice_gradient','dice_kdtree'):
        params['numerical_features'] = DS.dataset['numerical_features']
        params['categorical_features'] = DS.dataset['categorical_features']
        params['feature_names'] = DS.dataset['feature_names']

        clf = fit_model(X_train, X_test, Y_train, Y_test, params,DS=DS)
    ############
    elif 'NICE' in params['method']:
        logger.info(f"load dataset for nice")
        DS = load_dataset(dataset_name)
        X_train = DS.X_train
        X_test = DS.X_test
        Y_train = DS.y_train
        Y_test = DS.y_test
        X_prime = DS.X_counterfactual
        Y_prime = DS.y_counterfactual
        # if params['method'] in ('dice_gradient','dice_kdtree'):
        params['numerical_features'] = DS.dataset['numerical_features']
        params['categorical_features'] = DS.dataset['categorical_features']
        params['feature_names'] = DS.dataset['feature_names']

        clf = fit_model(X_train, X_test, Y_train, Y_test, params,DS=DS)
    
    else:
        DS = load_dataset(dataset_name)
        logger.info(f"load dataset for scfe")
        df, Y = load_data(dataname)
            # standardize data
        scaler = StandardScaler()
        X = df.values.astype(float)
        scaler.fit(X)
        X = scaler.transform(X)
        # normalize data rowwise by ell 2 norm
        X = normalize(X)
        # split data
        X_train, X_t, Y_train, Y_t = train_test_split(X, 
                                                        Y.astype(float), 
                                                        test_size=0.6666667,
                                                        random_state=10)

        # Make sure 1/3 is hold out data
        X_test, X_prime, Y_test, Y_prime = train_test_split(X_t, 
                                                            Y_t,
                                                            test_size=0.5,
                                                            random_state=11)

        ######################
        ###    FIT MODEL   ###
        ######################
        n_features = X_train.shape[1]
        params['categorical_features'] = []
        params['numerical_features'] = list(range(n_features))
        clf = fit_model(X_train, X_test, Y_train, Y_test, params)
        ######################
    # for dataname in params['datasets']:
        # print(f'Computing results on: {dataname}')
        # load and clean data
    
    # calculate losses if you want
    ##### We need LRT losses, not the original losses
    # train shadow models
    
    # load counterfactuals
    # calculate distances

    ########## ADDED TO REDUCE TEST RUNTIME< REMOVE WHEN FINALIZED
    # Subsample 20% of each set and pass the smaller samples to the pipeline
    # frac = 0.005
    # rs = params.get('random_state', 42)
    
    target_sample_size = params['attack_set_size']
    X_train_s, Y_train_s = subsample_data(
        X_train, Y_train, 
        target_size=target_sample_size,
        random_state=params.get('random_state', 42),
        stratify=True
    )
    X_test_s, Y_test_s = subsample_data(
        X_test, Y_test,
        target_size=target_sample_size,
        random_state=params.get('random_state', 42),
        stratify=True
    )
    X_prime_s, Y_prime_s = subsample_data(
        X_prime, Y_prime,
        target_size=target_sample_size,
        random_state=params.get('random_state', 42),
        stratify=True
    )
    
    

    logger.info(f"going to pipeline with subsamples")
    
    # Ensure all subsampled arrays have the same length by trimming larger ones to the smallest non-zero length.
    def _to_np(a):
        if a is None:
            return None
        return np.asarray(a)
    
    X_train_s = _to_np(X_train_s)
    X_test_s = _to_np(X_test_s)
    X_prime_s = _to_np(X_prime_s)
    Y_train_s = _to_np(Y_train_s)
    Y_test_s = _to_np(Y_test_s)
    Y_prime_s = _to_np(Y_prime_s)
    
    sizes = {
        'train': 0 if X_train_s is None else X_train_s.shape[0],
        'test': 0 if X_test_s is None else X_test_s.shape[0],
        'cf': 0 if X_prime_s is None else X_prime_s.shape[0]
    }
    
    # consider only positive sizes when computing the trim length
    positive_sizes = [s for s in sizes.values() if s > 0]
    if not positive_sizes:
        raise ValueError("All subsampled sets are empty; cannot proceed to pipeline.")
    
    min_size = min(positive_sizes)
    
    if any(s != min_size for s in sizes.values()):
        logger.info(f"Subsample sizes before trim: {sizes}; trimming to {min_size}")
        def _trim(X, y, n):
            if X is None or y is None:
                return X, y
            X = np.asarray(X)
            y = np.asarray(y)
            if X.shape[0] > n:
                X = X[:n]
                y = y[:n]
            return X, y
    
        X_train_s, Y_train_s = _trim(X_train_s, Y_train_s, min_size)
        X_test_s, Y_test_s = _trim(X_test_s, Y_test_s, min_size)
        X_prime_s, Y_prime_s = _trim(X_prime_s, Y_prime_s, min_size)
        logger.info(f"Subsample sizes after trim: train={X_train_s.shape[0]}, test={X_test_s.shape[0]}, cf={X_prime_s.shape[0]}")
    
    print(f"Using subsamples -> train: {X_train_s.shape[0]}, test: {X_test_s.shape[0]}, counterfactuals: {X_prime_s.shape[0]}")
    scores = pipeline(clf, X_train_s, X_test_s, X_prime_s, Y_train_s, Y_test_s, Y_prime_s, scores, params, dice_data=d,feature_names=DS.dataset['feature_names'])
    ##########

    # scores = pipeline(clf, X_train, X_test, X_prime, Y_train, Y_test, Y_prime, scores, params)

    return scores



def compute_curve(train_scores: list, test_scores: list,pos_label=1):
    '''
    Given scores for train and test:
    Ouput: tprs and fprs
    '''    
    y = np.r_[np.ones(np.shape(train_scores)[0]), np.zeros(np.shape(test_scores)[0])]
    fs, ts, thresholds = metrics.roc_curve(y, np.r_[train_scores, test_scores], pos_label=pos_label)
    return ts, fs


def plots(scores_train, scores_test, synth_gauss, params, method , runtime , title,seed, bins_train=100, bins_test=100, label=1):
    '''
    Given scores for train and test:
    Ouput: plots of log scaled AUC curves & distributions of stats
    '''
    plot_type = params['experiment']
    
    if plot_type == 'realworld':
        n_params =  [params['dataset']]
    else:
        n_params = synth_gauss['d']

    fig, axs = plt.subplots(1, 1 + len(n_params), figsize=(15, 3))
    
    # COMPUTE AUC CURVE
    all_tprs = []
    all_fprs = []
    
    for i in range(len(n_params)):
        tprs, fprs = compute_curve(scores_train[i], scores_test[i], label)
        all_tprs.append(tprs)
        all_fprs.append(fprs)
    
    for it, d in enumerate(n_params):
        auc = np.round(metrics.auc(all_fprs[it], all_tprs[it]), 2)
        if plot_type == 'synthetic':
            lab = str(d)
            axs[0].loglog(all_fprs[it], all_tprs[it], label=f"d={lab} - auc:{auc}")
        else:
            axs[0].loglog(all_fprs[it], all_tprs[it], label=f"{d} - auc:{auc}")
            
    axs[0].plot([0,1], [0,1], label='Random Baseline', linestyle='dotted', color='black')
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].set_xlim([0.001, 1.01])
    axs[0].set_ylim([0.001, 1.01])
    axs[0].set_title(title)
    axs[0].legend(framealpha=0.25)

    # COMPUTE DISTRIBUTION HISTOGRAMS
    for i in range(len(n_params)):
        if plot_type == 'synthetic':
            dim = n_params[i]
            n_train = scores_train[i].shape[0]
            eta = np.round(n_train / synth_gauss['d'][i], 2)
            print(f'(# train samples: {n_train}) / (dimension: {dim}) = eta: {eta}')
        axs[1+i].hist(scores_train[i], alpha=0.25, label='train', bins=bins_train, density=True)
        axs[1+i].hist(scores_test[i], alpha=0.25, label='test', bins=bins_test, density=True)
        axs[1+i].set_xlabel('Scores')
        axs[1+i].set_yscale('log')
        if plot_type == 'synthetic':
            axs[1+i].set_title(f'$\eta$ = {eta}')
        else:
            axs[1+i].set_title(f'{n_params[i]}')
        axs[1+i].legend()
    
    plt.tight_layout()
    n_shadow = params['n_shadow_models']
    n_ens = params['n_ensemble']
    experiment = params['experiment']
    dataset = params['dataset']
    attack_size = params['attack_set_size']
    shadow_frac = params['frac_ensemble']
    output_dir = f'./dpnice/experiments/{dataset}/{seed}/{method}/attack_size{attack_size}_shadowfrac{shadow_frac}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = f'{output_dir}{title}_{dataset}_nshadow{n_shadow}_nensemble{n_ens}.png'
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()
    


    # ###### added to extract metrics
    nmetrics = ["auc_roc", 'tpr_at_fpr_0', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1']

    # results = {}
    evaluator = classifier()

    # for col in score_df.columns:
        # scores = np.array(pd.to_numeric(score_df[col], errors='coerce'))
    scores_train = np.array(scores_train).flatten()
    scores_test = np.array(scores_test).flatten()
    scores = np.concatenate((scores_train, scores_test))
    true_labels = np.concatenate([
        np.ones(scores_train.shape[0], dtype=int),
        np.zeros(scores_test.shape[0], dtype=int)
    ])
    eval_result = evaluator.eval(true_labels, scores, use_decision_metrics=True)

    method_results = {}
    method_results['method'] = title
    for metric in nmetrics:
        method_results[metric] = eval_result.get(metric, np.nan)

    precision, recall, _ = precision_recall_curve(true_labels, scores)
    pr_auc = average_precision_score(true_labels, scores)
    method_results['pr_auc'] = pr_auc
    method_results['precision'] = precision.tolist()
    method_results['recall'] = recall.tolist()
    method_results['runtime'] = runtime

    return method_results
    

    # nmetrics = ["auc_roc", 'tpr_at_fpr_0', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1']
    # # scores = np.array(pd.to_numeric(score_df[col], errors='coerce'))
    # # y = np.r_[np.ones(np.shape(train_scores)[0]), np.zeros(np.shape(test_scores)[0])]
    # evaluator = AttackEvaluator(np.ones(np.shape(scores_train)), scores_train)
    # results = {}

    # if "roc" in nmetrics:
    #         target_fprs =  [0, 0.001, 0.01, 0.1]
    #         results.update(evaluator.roc_metrics(target_fprs=target_fprs))
        
    # if "classification" in nmetrics:
    #     decision_threshold = None
    #     results.update(evaluator.classification_metrics(decision_threshold=decision_threshold))
    
    # if "privacy" in nmetrics:
    #     decision_threshold = None
    #     results.update(evaluator.privacy_metrics(decision_threshold=decision_threshold))
    
    # if "epsilon" in nmetrics:
    #     confidence_level = 0.9
    #     threshold_method = "ratio"
    #     validation_split = 0.1
    #     results.update(evaluator.epsilon_evaluator(
    #     confidence_level=confidence_level, 
    #     threshold_method=threshold_method, 
    #     validation_split=validation_split)
    # )



    # eval_result_train =evaluator.eval(np.ones(np.shape(scores_train)), scores_train, use_decision_metrics=True)
    # eval_result_test = evaluator.eval(np.zeros(np.shape(scores_test)), scores_test, use_decision_metrics=True)
    # plt.savefig(f'{output_dir}{title}_{dataset}_nshadow{n_shadow}_nensemble{n_ens}.jpg')



synth_Dss = ['synth_adult','synth_hospital','synth_informs']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='heloc', help='heloc,acs_income,hospital,adult,informs,synth_adult,synth_informs,synth_hospital,compas,default_credit')
    parser.add_argument('--rseed', type=int, default=1, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='NN', help='NN, RF, XgBoost')
    parser.add_argument('--cf_method', type=str, default='dice_gradient', help='dice_kdtree, dice_gradient, NICE, scfe, cchvae, inline_DP')
    parser.add_argument('--attack_size', type=int, default=500, help='1500 3000 5000 10000')
    parser.add_argument('--shadow_frac', type=float, default=.75, help='0.1 ,0.15, 0.25, 0.5 1')
    
    logger.info("=" * 80)
    logger.info("STARTING CF_Distance_attack.py")
    logger.info("=" * 80)                                                                    
    
    args = parser.parse_args()
    dataset_name = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model
    CF_method = args.cf_method
    attack_size = args.attack_size
    shadow_frac = args.shadow_frac
    
    
    #### To do: assign settings from input arguments
    params = {
        'experiment': 'realworld',                                      # options: {'synthetic', 'realworld'}
        'dataset': dataset_name,  #['mnist', 'heloc', 'default', 'housing'],           # options: datasets: {'churn', 'mnist', 'heloc', 'default', 'housing'}
        'method': [CF_method],#CF_method,   'scfe',  dice_gradient,NICE                                       # current options for linear models: {'scfe','cchvae'}; for nonlinear models: {'scfe'}
        'fit_sgd': False,                                               # boolean: whether linear model should be fitted with sgd
        'linear': False, # True,                                                 # boolean: whether linear or nonlinear model should be fitted
        'hidden_layers': [100] * 1,                                     # list with ints: size of hidden layers (for nonlinear model)
        'epochs': 750,                                                  # int: number epochs (for nonlinear model)
        'batch_size': 32,                                               # int: batch size (for nonlinear model)
        'epsd': 1e-5,                                                   # pos real: constant to ensure stable loss evaluations
        'weighting': 'global',                                          # options: {'global', 'local'}: whether global variance or 1/n_shadow_models variance
        'n_shadow_models': 1,                                                                                     # int: number of shadow models
        'frac': 0.75,                                                   # frac: fraction of samples used for resampling scheme within shadow model train pipeline
        'penalty': 'none',                                             # options: {'none', 'l1', 'l2'}: reguralizer for underlying model
        'C': 1000,                                                      # pos real: regularization strength: higher leads less regularized models
        'ensemble': False,                                               # boolean: whether to fit an underlying ensemble model
        'n_ensemble': 1, # 250,                                              # int: number of ensemble models
        'frac_ensemble': shadow_frac, #0.75,                                          # pos real in [0,1]: fraction of samples used for resampling to train ensemble model
        'disjoint': True,                                               # boolean: Whether disjoint data sets should be used for training private ensemble
        'n_splits': 3,                                                  # int: Number of disjoint ensembles
        'lr_scfe': 0.005,                                               # pos. real in [0,1]:Learning rate for SCFE gradient descent
        'max_iter': 500,                                               # int: max number of iterations for SCFE gradient descent
        'quantile': 0.97,      # Number of splits for disjoint datasets
        'n_estimators': 100 ,
        'random_state': 42 ,
        'n_jobs': 5,
        'attack_set_size': attack_size,    ## the size of subsample for which attack is performed
        'max_depth': 10  # Number of splits for disjoint datasets
    }

# Parameters to control the synthetic data generating process described in the README.md
    synth_gauss = {    
        'n': 240,                              # n_train -> 5000
        'd': [50, 250, 1500, 1800, 3000],      # data dimensions
        'sigma2': 1,                           # data variance
        'frac_important_features': 0.1,        # fraction of relevant features
        'sigma2_eps': 0.01,                    # label noise
        'corr': 0.0,                           # correlation between features
    }
    
    
        
    #input: NICE_model,CF_method,to_explain,DS,filepathname,epsilon=1,NEIGHBOR_COUNT=3,is_synthetic = False, Real_DS = None
    resultsdir =    './dpnice/optimized/cf_results/{}/'.format(dataset_name)       
    modeloutdir = './dpnice/optimized/pretrained/{}/'.format(dataset_name)
    DSoutdir = './dpnice/optimized/datasets_loaded/{}/'.format(dataset_name)
    synth_DSoutdir = './dpnice/optimized/datasets_loaded/synth_{}/'.format(dataset_name)
    CF_file_dir = './dpnice/optimized/cf_files/{}/'.format(dataset_name)
    attack_dir = './dpnice/experiments/attack_results/{}/'.format(dataset_name)
    if dataset_name in synth_Dss:
        real_dataset_name = dataset_name.replace('synth_', '')
        Real_DSoutdir = './dpnice/optimized/datasets_loaded/{}/'.format(real_dataset_name)

    if dataset_name in synth_Dss : #generate model name, train mode and save model in the generated name, this name and training contains epsilon
        resultfilename = '{}{}_{}.csv'.format(resultsdir, model, RANDOM_SEED)
        model_name = '{}{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED)
        DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)
        real_DS_name = '{}{}{}.pkl'.format(Real_DSoutdir, model, RANDOM_SEED)
    else :  #generate model name, train mode and save model in the generated name, this name and training does not contain epsilon
        resultfilename = '{}{}_{}.csv'.format(resultsdir, model, RANDOM_SEED)
        model_name = '{}{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED)
        DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)
        synth_DS_name = '{}{}_{}.pkl'.format(synth_DSoutdir, model, RANDOM_SEED)
        attackfilename = '{}{}/cd_dist_attack_results.csv'.format(attack_dir, RANDOM_SEED)

    logger.info(f"parameters : {params}")
    ############
    shadows = [params['n_shadow_models']]
    ens = [params['n_ensemble']]
    for m in shadows:
        for n in ens:
            print(f'GENERATE RESULTS FOR: {m} SHADOW MODELS & {n} ENSEMBLE MODELS')
            params['n_shadow_models'] = m
            params['n_splits'] = params['n_ensemble'] = n
            logger.info(f"Going to perform experiments")
            scores = experiment(params,model_name,DS_name,  synth_gauss)
            # Generate plots: [log scaled AUC curves & train/test score distributions]
            results = {}
            results['losses'] = plots(scores['losses_train'], scores['losses_test'], synth_gauss, params, seed=RANDOM_SEED, runtime= scores['runtime_losses'],method='losses', title='Loss')
            
            results['stable_losses'] = plots(scores['stable_losses_train'], scores['stable_losses_test'], synth_gauss, params,seed=RANDOM_SEED,runtime=scores['runtime_stable_losses'],method='stable_losses', title='Stable loss')
            
            
            method_name = params['method'] if isinstance(params['method'], str) else params['method'][0]
            results['dist_lrt_global_'+method_name] = plots(scores['dists_lrt_train_global_' + method_name], scores['dists_lrt_test_global_' + method_name], synth_gauss, params, method_name,seed=RANDOM_SEED, runtime=scores['runtime_dists_lrt_' + method_name + '_global'],  title='Distance LRT (G. Var)', label=0)
            results['dist_lrt_local_'+method_name] = plots(scores['dists_lrt_train_local_' + method_name][0], scores['dists_lrt_test_local_' + method_name][0], synth_gauss, params, method_name ,seed=RANDOM_SEED,runtime=scores['runtime_dists_lrt_' + method_name + '_local'],title='Distance LRT (L. Var)', label=0)
            results['dists_'+method_name] = plots(scores['dists_train_' + method_name], scores['dists_test_' + method_name], synth_gauss, params, method_name,seed=RANDOM_SEED,runtime=scores['runtime_dists_' + method_name ], title='CF Distance')

            
            
            save_mia_results_to_csv(dataset_name,RANDOM_SEED,method_name,results,0,attack_size,output_file=attackfilename)


            all_rows = []
    


            # Save selected score arrays to CSV for downstream combined plots
            def _flatten(arr):
                return np.asarray(arr).flatten()

            f_losses_train = _flatten(scores['losses_train'])
            f_losses_test = _flatten(scores['losses_test'])
            f_stable_losses_train = _flatten(scores['stable_losses_train'])
            f_stable_losses_test = _flatten(scores['stable_losses_test'])
            d_train_global = _flatten(scores['dists_lrt_train_global_' + method_name])
            d_test_global = _flatten(scores['dists_lrt_test_global_' + method_name])
            d_train_local = _flatten(scores['dists_lrt_train_local_' + method_name][0]) if scores['dists_lrt_train_local_' + method_name] else np.array([])
            d_test_local = _flatten(scores['dists_lrt_test_local_' + method_name][0]) if scores['dists_lrt_test_local_' + method_name] else np.array([])
            d_train = _flatten(scores['dists_train_' + method_name])
            d_test = _flatten(scores['dists_test_' + method_name])

            max_len = max(len(d_train_global), len(d_test_global), len(d_train_local), len(d_test_local), len(d_train), len(d_test),len(f_losses_train), len(f_losses_test), len(f_stable_losses_train), len(f_stable_losses_test))
            def _pad(arr, n):
                if len(arr) < n:
                    arr = np.concatenate([arr, np.full(n - len(arr), np.nan)])
                return arr

            data = {
                'losses_train': _pad(f_losses_train, max_len),
                'losses_test': _pad(f_losses_test, max_len),
                'stable_losses_train': _pad(f_stable_losses_train, max_len),
                'stable_losses_test': _pad(f_stable_losses_test, max_len),
                'dists_lrt_train_global': _pad(d_train_global, max_len),
                'dists_lrt_test_global': _pad(d_test_global, max_len),
                'dists_lrt_train_local': _pad(d_train_local, max_len),
                'dists_lrt_test_local': _pad(d_test_local, max_len),
                'dists_train': _pad(d_train, max_len),
                'dists_test': _pad(d_test, max_len),
            }

            # save_dir = os.path.join('dpnice', 'experiments', dataset_name, method_name, f'seed_{RANDOM_SEED}')
            os.makedirs(attack_dir, exist_ok=True)
            # save_csv_path = os.path.join(save_dir, 'scores.csv')
            save_csv_path = '{}{}/{}/scores.csv'.format(attack_dir, RANDOM_SEED,method_name)
            sace_Dir = os.path.dirname(save_csv_path)
            if not os.path.exists(sace_Dir):
                os.makedirs(sace_Dir)
            pd.DataFrame(data).to_csv(save_csv_path, index=False)

            print(scores)

import multiprocessing, psutil, time

def print_affinity():
    try:
        affinity = psutil.Process().cpu_affinity()
        print("Affinity:", affinity)
    except AttributeError:
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        if logical:
            # If physical count is available, prefer range over logical count
            if physical and physical > 0:
                affinity = list(range(physical))
                print("Affinity (fallback from physical cores):", affinity)
            else:
                print("Affinity (fallback logical cpu count):", list(range(logical)))
        else:
            print("Could not determine CPU affinity or counts.")


print("CPUs available:", multiprocessing.cpu_count())
print_affinity()

