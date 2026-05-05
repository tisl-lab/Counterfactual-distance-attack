import argparse
import os
# import random
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dice_ml
from utils.data.load_data import Fetcher
import tensorflow as tf
from dpmechanisms.new_nice import NICE
from tools.new_models import create_model_pipeline, calculate_accuracy, load_best_params


def _dice_encode(d, X: pd.DataFrame) -> np.ndarray:
    """Encode + scale with DiCE (OHE + MinMax), version-agnostic."""
    try:
        X_enc = d.get_ohe_min_max_normalized_data(X)
        return np.asarray(X_enc, dtype=np.float32)
    except Exception:
        pass
    try:
        X_enc = d.transform_data(X, encode=True, scale=True)
        return np.asarray(X_enc, dtype=np.float32)
    except Exception:
        pass
    try:
        X_enc = d.transform_data(X, _encode=True, _scaled=True)
        return np.asarray(X_enc, dtype=np.float32)
    except Exception:
        pass
    raise RuntimeError("Failed to encode via DiCE; check dice-ml version.")


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
    

class PredictProbaAdapter:
    """Sklearn-style predict_proba adapter that accepts numpy arrays or DataFrames."""
    def __init__(self, clf, feature_names):
        self.clf = clf
        self.feature_names = list(feature_names)

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self.feature_names)
        return self.clf.predict_proba(df)


def save_accuracy_csv(accuracy: float, dataset_name: str, cf_method: str, rseed: int, out_path: str = "dpnice/mia_model_accuracy/metrics.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    row = pd.DataFrame([{
        "dataset_name": dataset_name,
        "cf_method": cf_method,
        "rseed": rseed,
        "accuracy": float(accuracy)
    }])
    # Append (no header if file exists)
    header = not os.path.exists(out_path)
    row.to_csv(out_path, mode="a", header=header, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="compas",help="Dataset name: adult, heloc, informs, synth_adult")
    # parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-cfs", type=int, default=1)
    parser.add_argument("--rseed", type=int, default=0)
    parser.add_argument("--cf_method", type=str, default="NICE", help="CF method: dice_kdtree, dice_gradient, NICE")
    parser.add_argument("--prox-weight", type=float, default=0.5, help="Proximity loss weight")
    parser.add_argument("--div-weight", type=float, default=0.1, help="Diversity loss weight")
    parser.add_argument("--cf-lr", type=float, default=0.1, help="Learning rate for CF optimization")
    # parser.add_argument("--cf-optimizer", type=str, default="pytorch:adam", help="CF optimizer (pytorch:adam/sgd)")

    args = parser.parse_args()
    # main_dice_generic(args)
    dataset_dir =f'./dpnice/mia_dataset_loaded/{args.dataset_name}/'
    explainer_dir = f'./dpnice/mia_explainer/{args.dataset_name}/{args.cf_method}/'
    DS_name = '{}seed_{}.pkl'.format(dataset_dir, args.rseed)
    explainer_name = '{}seed_{}.pkl'.format(explainer_dir,args.rseed)
    
    if not os.path.exists(explainer_dir):
        os.makedirs(explainer_dir)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    DS = load_dataset(args.dataset_name)
    # dill.dump(DS, open(f"Dataset_{args.dataset_name}_{args.rseed}.pkl", "wb"))
    dill.dump(DS, open(DS_name,"wb"))
    
    if args.cf_method == "dice_gradient":
        
        train_df = DS.X_train.copy()
        train_df["target"] = DS.y_train
        test_df = DS.X_test.copy()
        test_df["target"] = DS.y_test
        continuous = [DS.dataset['feature_names'][i] for i in DS.dataset['numerical_features']]
        d = dice_ml.Data(dataframe=train_df , continuous_features=continuous, outcome_name="target")

        # Encode using DiCE (OHE + MinMax); encoded width adapts to dataset
        # Xtr_enc = _dice_encode(d, DS.X_train)
        # Xte_enc = _dice_encode(d, DS.X_test)
        
        
        Xtr_df = d.get_ohe_min_max_normalized_data(DS.X_train)
        Xte_df = d.get_ohe_min_max_normalized_data(DS.X_test)
        Xte_df = Xte_df.reindex(columns=Xtr_df.columns, fill_value=0.0)
        Xtr_enc = np.asarray(Xtr_df, dtype=np.float32)
        Xte_enc = np.asarray(Xte_df, dtype=np.float32)
        enc_dim = Xtr_enc.shape[1]
        print(f"[DiCE] Encoded shapes -> train: {Xtr_enc.shape}, test: {Xte_enc.shape}")


        # model = SimpleANN(input_dim=x_train.shape[1], hidden_units=20, output_dim=1)
        # Define model for binary classification
        input_dim = Xtr_enc.shape[1]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),  # for binary classification
            metrics=['accuracy']
        )

        model.fit(Xtr_enc, DS.y_train, epochs=30, batch_size=16)    
        
        # Evaluate accuracy on test (encoded)
        test_pred = (model.predict(Xte_enc) > 0.5).astype(int).flatten()

        acc = (test_pred == np.asarray(DS.y_test)).mean()
        print(f"Model test accuracy: {acc:.4f}")
        # SAVE accuracy
        save_accuracy_csv(acc, args.dataset_name, args.cf_method, args.rseed)

        backend = 'TF2'  # needs tensorflow installed
        m = dice_ml.Model(model=model, backend=backend, func="ohe-min-max")
        # m = dice_ml.Model(model_type='classifier', model=m, backend=backend)
        exp = dice_ml.Dice(d, m, method="gradient")

        # generate counterfactuals
        # dice_exp = exp.generate_counterfactuals(DS.X_test[1:10], total_CFs=1, desired_class="opposite")
        # visualize the result, highlight only the changes
        # dice_exp.visualize_as_dataframe(show_only_changes=True)
        dill.dump(exp,open(explainer_name,"wb"))
    
    elif args.cf_method in ["NICE","dice_kdtree"]: 
        params = {
            "activation": "relu",
            "alpha": 0.001,
            "hidden_layer_sizes": [20],
            "learning_rate": "constant",
            "solver": "adam"
        }
        
        clf = create_model_pipeline("NN",params,DS.dataset['numerical_features'],DS.dataset['categorical_features'])
        ## NN
        
        if args.cf_method == "dice_kdtree":
            clf.fit(DS.X_train, DS.y_train)
            train_df = pd.DataFrame(DS.X_train, columns=DS.dataset['feature_names'])
            train_df['target'] = DS.y_train
            prediction = lambda x: clf.predict_proba(x)

            #Note: calculate model accuracy on test data
            acc = calculate_accuracy(clf, DS.X_test, DS.y_test)
            print(f"Model test accuracy: {acc}")    
            save_accuracy_csv(acc, args.dataset_name, args.cf_method, args.rseed)
            #### update
            train_df = DS.X_train.copy()
            train_df["target"] = DS.y_train
            test_df = DS.X_test.copy()
            test_df["target"] = DS.y_test
            continuous = [DS.dataset['feature_names'][i] for i in DS.dataset['numerical_features']]
            d = dice_ml.Data(dataframe=train_df , continuous_features=continuous, outcome_name="target")
            
            backend = 'sklearn'  # or 'pytorch' or 'TF2'
            m = dice_ml.Model(model=clf, backend=backend)

            exp = dice_ml.Dice(d, m, method="kdtree")
            # dice_exp = exp.generate_counterfactuals(DS.X_test[1:10], total_CFs=1, desired_class="opposite")
            # dice_exp.visualize_as_dataframe(show_only_changes=True)
            dill.dump(exp,open(explainer_name,"wb"))
            # dill.dump(exp, open(f"DiCE_kdtree_exp_{args.dataset_name}_{args.rseed}.pkl", "wb"))
        elif args.cf_method == "NICE":
            clf.fit(DS.X_train.values, DS.y_train)
            
            feature_names = DS.dataset['feature_names']
            prediction = PredictProbaAdapter(clf, feature_names)
            # prediction = lambda x: clf.predict_proba(pd.DataFrame(x, columns=DS.dataset['feature_names'])) if not isinstance(x, pd.DataFrame) else clf.predict_proba(x)
            # prediction = lambda x: clf.predict_proba(x)

            #Note: calculate model accuracy on test data
            acc = calculate_accuracy(clf, DS.X_test, DS.y_test)
            print(f"Model test accuracy: {acc}")   
            save_accuracy_csv(acc, args.dataset_name, args.cf_method, args.rseed)
            NICE_model = NICE(optimization='proximity', #'differentialprivacy',  #'plausibility',    #optimization method
                    justified_cf=True,
                    X_train = DS.X_train.values,   #training data
                    predict_fn=prediction,    #prediction function
                    y_train = DS.y_train,
                    #if optimization = 'Plausibility'
                    #   auto_encoder= autoencoder,
                        #if optimization is 'differentialprivacy'
                        # set k_neighbors
                    #categorical features 
                    cat_feat=DS.dataset['categorical_features'],
                    #numerical features   
                    num_feat=DS.dataset['numerical_features']) 
            dill.dump(NICE_model,open(explainer_name,"wb"))
            # dill.dump(NICE_model, open(f"NICE_exp_{args.dataset_name}_{args.rseed}.pkl", "wb"))
            # CF,_ = NICE_model.explain_Second(DS.X_test.values[0:1])
            # print("Original Instance: ", DS.X_test.values[0:1])
            # print("Counterfactuals: ", CF)
    else:   
        print(f"CF method {args.cf_method} not recognized.")
