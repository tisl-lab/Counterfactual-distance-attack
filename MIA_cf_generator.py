import pandas as pd
import argparse
import dill
import copy
import sys
import time
from tools.save_results import generate_CF_record,save_CF_to_csv,add_to_CF_file
from dpmechanisms.local_dp import Make_Private,Compare,make_DS_LDP
from utils.analysis.evaluation import reidentify,compare_solutions
from utils.data.distance import KLDivergence
from mpi4py import MPI
import numpy as np
from utils.data.distance import heom_distance_new

import dice_ml
import tensorflow as tf

def split(container, count):
    return [container[_i::count] for _i in range(count)]

def generate_CF(Explainer,CF_method,instance_num,DS,epsilon=1,n_count=3,is_synthetic = False, Real_DS = None,is_dp_ds = False,index = 0,iter = 0):
    to_explain = DS.X_counterfactual[instance_num:instance_num+1]
    
    if CF_method == 'NICE':
        if(epsilon == 1 or is_synthetic == True):
            if(is_synthetic == False):
                X_train = DS.X_train
            else: # in case we used synthetic data to train the model and we want to explain real data and find reidentification rate for real data
                X_train = Real_DS.X_train
            prediction = Explainer.data.predict_fn(to_explain).argmax() 
            t0 = time.perf_counter()
            CF,basic_instance = Explainer.explain_Second(to_explain.values)
            cf_time_sec = time.perf_counter() - t0
            # add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember=True)
            # add_to_CF_file(NICE_CF_filename,CF[0], columns_list=DS.dataset['feature_names'])
                ## if our private countefactual is yet a counterfactual
            nice_df = pd.DataFrame(data=[to_explain.values[0], CF[0]], columns=DS.dataset['feature_names'])
            
            if (Explainer.data.predict_fn(CF).argmax() == prediction):
                no_CF = True 
                print("for epsilon:",epsilon,"NICE:",CF,"is not Counterfactual")
                record = generate_CF_record(CF_method, nice_df, CF_distance='NA', CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' ,cf__ri_count='NA', not_cf=no_CF, epsilon=0, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA',index=index,iter=iter,time_sec=cf_time_sec)
                # save_CF_to_csv(record, csv_file_path=resultfilename)
                ## added to save all CFs in cf_file
                cf__ri_count =reidentify(CF,X_train)
                if cf__ri_count > 0 :
                    member = True
                else: 
                    member = False
                return member,record,None,None,basic_instance
            else:
                no_CF = False
                plaus_distances = Explainer.Calculate_Plaus_Dists(CF)
                CF_distance = Explainer.distance_metric.measure(CF,to_explain.values[0])
                print("for epsilon:",epsilon, "NICE:",nice_df,"Distance:",CF_distance)
                # eq_to_basic = Compare(CF,ldp_cf)  #if original instance has been changed by ldp or not
                # changes_rate, same_inst_rate = compare_solutions(to_explain,CF,ldp_cf)
                cf__ri_count =reidentify(CF,X_train)
                # nice_KL_divergence = KL_divergence_machine.distance(basic_instance,CF)
                record = generate_CF_record(CF_method, nice_df, CF_distance,plaus_distances[0],plaus_distances[1],plaus_distances[2], cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', same_inst_rate='NA',SameAsNice= 'NA',index=index,iter=iter)
                # record = generate_CF_record(CF_method, nice_df, CF_distance,plaus_distances[0],plaus_distances[1],plaus_distances[2], cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', same_inst_rate='NA',SameAsNice= 'NA',KL_divergence=nice_KL_divergence)
                # save_CF_to_csv(record, csv_file_path=resultfilename)
                if cf__ri_count > 0 :
                    member = True
                else: 
                    member = False
                # add_to_CF_file(CF_filename,CF[0], columns_list=DS.dataset['feature_names'],ismember=member)
            
                return member,record,CF,None,basic_instance

    
    
 ##############
    elif CF_method in ['dice','dice_genetic','dice_kdtree','dice_gradient']:
    # elif CF_method == 'dice':
            dice_exp = Explainer # to keep consistency we call all models NICE_model from beginning
            
            for col in to_explain.columns:
                # to_explain.loc[:, col] = to_explain[col].astype('float32')
                try:
                    if CF_method == 'dice_gradient':
                        t0 = time.perf_counter()
                        dice_cf = dice_exp.generate_counterfactuals(to_explain, total_CFs=1, desired_class="opposite")
                        cf_time_sec = time.perf_counter() - t0
                    elif CF_method == 'dice_kdtree':
                        X_df = pd.DataFrame(to_explain, columns=DS.dataset['feature_names'])
                        t0 = time.perf_counter()
                        dice_cf = dice_exp.generate_counterfactuals(X_df, total_CFs=1, desired_class="opposite")
                        cf_time_sec = time.perf_counter() - t0
            
                    cf_df = dice_cf.cf_examples_list[0].final_cfs_df
                    
                    for _, cf in cf_df.iterrows():
                        # add_to_CF_file(CF_filename, cf, columns_list=DS.dataset['feature_names'])
                        no_CF = False
                        print("for instance:",instance_num, "run", CF_method )
                        ##### update to use dice model distance metric for this part
                        # Use DiCE's distance metric to calculate distance between cf and to_explain
                        # Drop 'target' column from cf if it exists
                        if 'target' in cf.index or 'target' in cf.columns:
                            cf_class = int(cf['target'])  # Assuming binary classification with classes 0 and 1
                            cf = cf.drop('target')
                            
                        plause_dists = []
                        cf_np = cf.values.reshape(1, -1)
                        to_explain_np = to_explain.values
                        
                        
                        CF_distance = heom_distance_new(to_explain_np[0], cf_np[0],DS.dataset['numerical_features'],DS.dataset['feature_ranges'])

                        
                        distances = [heom_distance_new(x.values.reshape(1, -1)[0], cf_np[0],DS.dataset['numerical_features'],DS.dataset['feature_ranges']) for _, x in DS.X_train.iterrows()]
                        closest_idx = DS.X_train.index[np.argmin(distances)]
                        ##### calculate plaus distances
                        #  1. Distance to the closest instance in the training set
                        closest_instance =DS.X_train.loc[closest_idx]
                        plause_dists.append(heom_distance_new(closest_instance.values.reshape(-1,1)[0], cf_np[0],DS.dataset['numerical_features'],DS.dataset['feature_ranges']))
                        #  2. Avg distance to KNN instances to CF class
                        # Find k nearest neighbors to the counterfactual in the counterfactual class from training dataset
                        k = 5  # or set as needed, or pass as argument
                        # Set cf_class to the class predicted by dice_exp for cf
                        # cf_class = int(dice_exp.predict_fn(cf.values.reshape(1, -1)).argmax())
                        # if cf_class is not None:
                        # X_train_cf_class = DS.X_train[dice_exp.predict_fn(DS.X_train.values).argmax(axis=1) == cf_class]
                        if CF_method == 'dice_genetic':
                            encoded_x_train = dice_exp.label_encode (DS.X_train)
                            preds = dice_exp.predict_fn(encoded_x_train)
                            # preds = dice_exp.predict_fn(DS.X_train.values)
                        elif CF_method in ['dice_kdtree','dice_gradient']:
                            preds = dice_exp.predict_fn(DS.X_train)
                        # else:
                        #     preds = dice_exp.predict_fn(DS.X_train)
                        print(preds.shape)  # Debug: see what you get

                        if preds.ndim == 2:
                            pred_classes = preds.argmax(axis=1)
                        else:
                            pred_classes = preds  # Already class labels
                        X_train_cf_class = DS.X_train.loc[pred_classes == cf_class]
                        # X_train_cf_class = DS.X_train.loc[dice_exp.predict_fn(DS.X_train.values).argmax(axis=1)==cf_class]
                        # X_train_cf_class = DS.X_train[dice_exp.predict_fn(DS.X_train.values).argmax(axis=1) == cf_class]
                        # else:
                            # X_train_cf_class = X_train
                        if len(X_train_cf_class) >= k:
                            # Compute HEOM distances between cf and all instances in X_train_cf_class
                            cf_np = cf.values.reshape(1, -1)
                            X_train_cf_class_np = X_train_cf_class.values
                            heom_distances = [
                                heom_distance_new(x[1].values.reshape(1, -1)[0], cf_np[0], DS.dataset['numerical_features'], DS.dataset['feature_ranges'])
                                for x in X_train_cf_class.iterrows()
                            ]
                            # heom_distances = heom_distance_new(x.values.reshape(1, -1)[0], cf_np[0],DS.dataset['numerical_features'],DS.dataset['feature_ranges']) for  x in X_train_cf_class.iterrows()
                            
                            # knn_indices = X_train_cf_class.index[np.argsort(heom_distances)[:k]]
                            # knn_instances = X_train_cf_class.loc[knn_indices]
                            avg_knn_heom_dist = np.mean(np.array(heom_distances)[np.argsort(heom_distances)[:k]])
                            # avg_knn_dist = knn_instances.apply(lambda row: dice_exp.distance.compute_dist(row, cf), axis=1).mean()
                        else:
                            avg_knn_heom_dist = float('nan')
                        plause_dists.append(avg_knn_heom_dist)

                        #  3. Avg. distance to a random subset of instances in CF class
                        if len(X_train_cf_class) > 0:
                            rand_k = min(k, len(X_train_cf_class))
                            rand_indices = np.random.choice(X_train_cf_class.index, size=rand_k, replace=False)
                            rand_instances = X_train_cf_class.loc[rand_indices]
                            avg_rand_heom_dist = rand_instances.apply(
                                lambda row: heom_distance_new(row.values.reshape(1, -1)[0], cf_np[0], DS.dataset['numerical_features'], DS.dataset['feature_ranges']),
                                axis=1
                            ).mean()
                            # plause_dists.append(avg_rand_heom_dist)
                            # avg_rand_dist = rand_instances.apply(lambda row: dice_exp.distance.compute_dist(row, cf), axis=1).mean()
                        else:
                            avg_rand_heom_dist = float('nan')
                        plause_dists.append(avg_rand_heom_dist)
                    
                        # return plause_dists
                        #############


                        #### should update
                        eq_to_basic = float('nan') #Compare(basic_instance,ldp_ds_cf)  #if original instance has been changed by ldp or not
                        changes_rate, same_inst_rate = float('nan')  ,float('nan') # compare_solutions(to_explain,basic_instance,ldp_ds_cf)
                        cf__ri_count =reidentify(cf_np,DS.X_train)
                        # ldp_ds_cf_KL_divergence = KL_divergence_machine.distance(basic_instance,ldp_ds_cf)
                        dice_df = pd.DataFrame(data=[to_explain.values.flatten(), cf.values.flatten()],columns=cf.index
)
                        record = generate_CF_record(CF_method, dice_df, CF_distance, CF_min_dist= plause_dists[0],CF_min_k_dist=plause_dists[1],CF_rand_k_dist=plause_dists[2], cf__ri_count= cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,index=index,iter=iter,time_taken=cf_time_sec)
                                    # record = generate_CF_record(CF_method, ldp_ds_df, CF_distance, CF_min_dist= plaus_distances[0],CF_min_k_dist=plaus_distances[1],CF_rand_k_dist=plaus_distances[2], cf__ri_count= cf__ri_count, not_cf=no_CF, epsilon=epsilon, k=0, changed_rate=changes_rate, same_inst_rate=same_inst_rate,SameAsNice= eq_to_basic,KL_divergence=ldp_ds_cf_KL_divergence)
                        # save_CF_to_csv(record, csv_file_path=resultfilename)
                        
                        # add_to_CF_file(CF_filename,ldp_ds_cf[0], columns_list=DS.dataset['feature_names'],ismember=(cf__ri_count>0))
                        # add_to_CF_file(Basic_instances_filename,basic_instance[0], columns_list=DS.dataset['feature_names'],ismember= (reidentify(basic_instance,X_train)>0))
                        if cf__ri_count > 0 :
                            member = True
                        else: 
                            member = False
                        return member,record,cf_np,None,None

                except Exception as e:
                    print(f"Failed to generate counterfactuals for instance {to_explain}: {e}")
                    no_CF = True 
                    dice_df = pd.DataFrame(data=[to_explain.iloc[0]], columns=DS.dataset['feature_names'])
                    empty_df = pd.DataFrame(columns=DS.dataset['feature_names'])
                    record = generate_CF_record(CF_method, empty_df, CF_distance='NA', CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA' ,cf__ri_count='NA', not_cf=no_CF, epsilon=epsilon, k=0, changed_rate='NA', SameAsNice='NA',same_inst_rate='NA',index=index,iter=iter)
                    # save_CF_to_csv(record, csv_file_path=resultfilename)
                    
                    cf__ri_count =0
                    if cf__ri_count > 0 :
                        member = True
                    else: 
                        member = False
                    return member,record,"NAN",None,None
    else:
        print("CF method not implemented yet.")
        return None,None,None,None,None
##############

def main(args):
    dataset_name = args.dataset_name
    RANDOM_SEED = args.rseed
    # model = args.model
    # epsilon = args.epsilon
    CF_method = args.cf_method
    # n_count = args.n_count
    INS_COUNT =args.INS_COUNT                                                          
    # instance_num = args.instance_num
    model = "NN"
    # n_count = args.n_count
    


    
    #input: NICE_model,CF_method,to_explain,DS,filepathname,epsilon=1,NEIGHBOR_COUNT=3,is_synthetic = False, Real_DS = None
    dataset_dir =f'./dpnice/mia_dataset_loaded/{args.dataset_name}/'
    explainer_dir = f'./dpnice/mia_explainer/{args.dataset_name}/{args.cf_method}/'
    resultsdir =    './dpnice/mia_explainer/cf_results/{}/{}/'.format(dataset_name,CF_method)       
    # modeloutdir = './dpnice/mia_explainer/pretrained/{}/'.format(dataset_name)
    CF_file_dir = './dpnice/mia_explainer/cf_files/{}/{}/'.format(dataset_name,CF_method)
    
    DS_name = '{}seed_{}.pkl'.format(dataset_dir, args.rseed)
    explainer_name = '{}seed_{}.pkl'.format(explainer_dir,args.rseed)
    resultfilename = '{}seed_{}.csv'.format(resultsdir, args.rseed)
    CF_filename =  '{}seed_{}.csv'.format(CF_file_dir,args.rseed)
    Basic_instances_filename =  '{}{}/seed_{}_basic.csv'.format(CF_file_dir,CF_method,args.rseed)
    # NICE_CF_filename = '{}{}_{}_eps_{}.csv'.format(CF_file_dir,model,RANDOM_SEED,epsilon)
    
    
    
    
    # 'NICE','zerocost_DP_CF','ldp_ds_cf','synth_dp_cf','LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP'
    
        #make DS ldped
    
    gen_c = False
    is_dp_ds = False
    is_synthetic = False
    is_synth_dp_ds = False
    

    if (CF_method in ('NICE','dice_gradient','dice_kdtree')):
        #load the dataset
        with open(DS_name, 'rb') as file:
            DS = dill.load(file)
                # print("DS loaded")

        # if CF_method == 'dice_gradient':
            
        #         train_df = DS.X_train.copy()
        #         train_df["target"] = DS.y_train
        #         test_df = DS.X_test.copy()
        #         test_df["target"] = DS.y_test
        #         continuous = [DS.dataset['feature_names'][i] for i in DS.dataset['numerical_features']]
        #         d = dice_ml.Data(dataframe=train_df , continuous_features=continuous, outcome_name="target")

        #         Xtr_df = d.get_ohe_min_max_normalized_data(DS.X_train)
        #         Xte_df = d.get_ohe_min_max_normalized_data(DS.X_test)
        #         Xte_df = Xte_df.reindex(columns=Xtr_df.columns, fill_value=0.0)
        #         Xtr_enc = np.asarray(Xtr_df, dtype=np.float32)
        #         Xte_enc = np.asarray(Xte_df, dtype=np.float32)
        #         enc_dim = Xtr_enc.shape[1]
        #         print(f"[DiCE] Encoded shapes -> train: {Xtr_enc.shape}, test: {Xte_enc.shape}")


        #         input_dim = Xtr_enc.shape[1]

        #         model = tf.keras.Sequential([
        #             tf.keras.layers.Input(shape=(input_dim,)),
        #             tf.keras.layers.Dense(20, activation='relu'),
        #             tf.keras.layers.Dense(1, activation='sigmoid')
        #         ])

        #         # Compile model
        #         model.compile(
        #             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #             loss=tf.keras.losses.BinaryCrossentropy(),  # for binary classification
        #             metrics=['accuracy']
        #         )

        #         model.fit(Xtr_enc, DS.y_train, epochs=30, batch_size=16)    
                
        #         # Evaluate accuracy on test (encoded)
        #         # test_pred = (model.predict(Xte_enc) > 0.5).astype(int).flatten()

        #         # acc = (test_pred == np.asarray(DS.y_test)).mean()
        #         # print(f"Model test accuracy: {acc:.4f}")
        #         # SAVE accuracy
        #         # save_accuracy_csv(acc, args.dataset_name, args.cf_method, args.rseed)

        #         backend = 'TF2'  # needs tensorflow installed
        #         m = dice_ml.Model(model=model, backend=backend, func="ohe-min-max")
        #         # m = dice_ml.Model(model_type='classifier', model=m, backend=backend)
        #         Explainer = dice_ml.Dice(d, m, method="gradient")
            
        # else:
        with open(explainer_name, 'rb') as file:
            Explainer = dill.load(file)
            # print("Model loaded")
            

        gen_c = True
        # print("call CF generator")
        params = {'Explainer' : Explainer,'CF_method':CF_method,'DS': DS,'filepathname' : resultfilename}
    
    # elif (dataset_name not in synth_Dss and  CF_method in ('dice','dice_genetic','dice_kdtree')):
        
    
    else:
        gen_c = False
        print("CF method not implemented or supported yet.")
        sys.exit()


        # if you are main process, recieve and save all to file
        # if you are not the main process, send result to the main process
    if gen_c == True:
    # Generate list of indexes for which we generate CFs
    # if model prediction and true label are false then add it to instance_lists    
        indices = []
        instance_num = 0
        idx = 0
        while True:
        # for instance_num in range(1,101):
            if instance_num == INS_COUNT or idx>= DS.X_counterfactual.shape[0]: 
                if idx>= DS.X_counterfactual.shape[0]:
                    print("Not enough instances to generate CFs, Only {} instances are available".format(DS.X_counterfactual.shape[0]))
                break
            else:
                # to_explain = DS.X_counterfactual[idx:idx+1 :] 
                # if NICE_model.data.predict_fn(to_explain).argmax() == 0:  # if model prediction is not desired 
                ### we want to generate a semi balances attack dataset, so we do not need for our instances to be in one class, we prefer diversity
                indices.append(idx)
                
                instance_num +=1
                idx +=1

        # indices = [x for x in range(0,100)]
        # Start parallelisation.
        
        COMM = MPI.COMM_WORLD
        size = COMM.Get_size()
    # the root process splits jobs (# do paralleling over instances)
        if COMM.rank == 0:
            jobs = split(indices, COMM.size)
            print("process {} splitted jobs:",COMM.rank," Jobs: ", jobs)
        else:
            jobs = None

        # the root process scatters jobs
        jobs = COMM.scatter(jobs, root=0)

        results = []
        for job in jobs:
            names = ['membership','cf_record', 'CF', 'nice_cf','basic_cf']
            # cf_record,CF,nice_cf,basic_cf = generate_cf_Parallel(NICE_model,CF_method,job,DS,epsilon,NEIGHBOR_COUNT,is_synthetic = False)
            for iter in range(0,1):
                cf_dict = dict(zip(names, generate_CF(Explainer,CF_method,job,DS,index = job, iter=iter))) ### Here we only generated non synthetic CFs, work on synthetic CFs and DP Ds
                
            ### Make result a dictionart instead of a list
            #############
            ##### add all returned material to the dict
            
            #add results to result
                results.append(cf_dict)

            
        # print("from process:",COMM.rank," Jobs: ", jobs)
        # return result
        # COMM.Barrier() 
        results = MPI.COMM_WORLD.gather(results, root=0)

        # file_names = ["file_1", "file_2", "file_3"]
        # CF_filename =  '{}{}seed_{}.csv'.format(CF_file_dir,CF_method,RANDOM_SEED)
        Basic_instances_filename =  '{}{}seed_{}_basic.csv'.format(CF_file_dir,CF_method,RANDOM_SEED,)
        # NICE_CF_filename = '{}{}seed_{}_{}_eps_{}_k_{}_NICE.csv'.format(CF_file_dir,model,RANDOM_SEED,CF_method,epsilon,n_count)

        if(COMM.rank == 0):

            results = [_i for temp in results for _i in temp]
            # process dict as a whole, save all 3/4 files
            
            for idx, objects in enumerate(results):
                
                for name, obj in objects.items():
                    if(name == 'membership'):
                        membership = obj
                    elif(name =='cf_record'):
                        # filename = f"{file_names[idx]}_{name}.txt"
                        save_CF_to_csv(record=obj,csv_file_path=resultfilename)
                    elif(name == 'CF' and (obj is not None)): 
                        add_to_CF_file(CF_filename,obj[0], columns_list=DS.dataset['feature_names'],ismember=membership)
                    # elif(name == 'nice_cf' and (obj is not None)):
                    #     add_to_CF_file(NICE_CF_filename,obj[0], columns_list=DS.dataset['feature_names'])
                    # elif(name == 'basic_cf' and (obj is not None)):
                        # add_to_CF_file(Basic_instances_filename,obj[0], columns_list=DS.dataset['feature_names'],ismember= True)



            # for idx,object in results:
            #     save_CF_to_csv(record=record,csv_file_path=resultfilename)
            
        if COMM.rank == 0:        
            MPI.Finalize()


    # generate_CF_record()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset_name', type=str, default='compas', help='hospital,adult,informs,synth_adult,synth_informs,synth_hospital,compas,default_credit')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 5')
    # parser.add_argument('--model', type=str, default='NN', help='NN, RF, XgBoost')
    # parser.add_argument('--n_count', type=int, default=3, help='3 5 10 20')
    # parser.add_argument('--epsilon', type=float, default='1', help='.01, .1, 1, 5, 10')
    parser.add_argument('--cf_method', type=str, default='dice_gradient', help='NICE,dice_kdtree,zerocost_DP_CF,ldp_ds_cf,synth_dp_cf,LDP_CF,LDP_Noisy_max,inline_LDP,LDP_SRR,LDP_Noisy_max')
    parser.add_argument('--INS_COUNT', type=int, default=10000, help='random seed: choose between 0 - 5')

    # get input      
    args = parser.parse_args()
    main(args)

