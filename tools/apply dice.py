import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers
from utils.data.load_data import Fetcher

# Existing code...

# def generate_counterfactuals_dice(model, dataset, instance, num_counterfactuals=5):
#     # Initialize DiCE
#     d = Dice(model, method="random")
    
#     # Generate counterfactuals
#     dice_exp = d.generate_counterfactuals(instance, total_CFs=num_counterfactuals, desired_class="opposite")
    
#     # Convert counterfactuals to DataFrame
#     cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    
#     return cf_df

# # Example usage
# # if epsilon >= 1: # to open saved models and datasets correctly based on their names
# #     epsilon = int(epsilon)

# # Assuming you have a trained model and a dataset
# model = 'RF'  # Your trained model
# dataset = 'acs_income'  # Your dataset
# instance = 1  # An instance for which you want to generate counterfactuals

# # Generate counterfactuals using DiCE
# cf_dice = generate_counterfactuals_dice(model, dataset, instance)

# # Print the generated counterfactuals
# print(cf_dice)

import dice_ml
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split

dataset = helpers.load_adult_income_dataset()
target = dataset["income"] # outcome variable
train_dataset, test_dataset, _, _ = train_test_split(dataset,
                                                     target,
                                                     test_size=0.2,
                                                     random_state=0,
                                                     stratify=target)
# Dataset for training an ML model

DS = Fetcher(dataset) #hospital,adult,informs,synth_adult
X= DS.dataset['X'].values
Y = DS.dataset['y'].values
#Create pipeline

X_train, X_test_1, y_train, y_test_1 = train_test_split(X, Y, test_size=0.4,random_state=RANDOM_SEED)
# here: select 1% of the dataset to check for exponential mechanism


DS.X_train = X_train.copy()
DS.y_train = y_train.copy()

X_test,X_counterfactual,y_test,y_counterfactual=train_test_split(X_test_1, y_test_1, test_size=0.5,random_state=RANDOM_SEED)
DS.X_test = X_test.copy()
DS.y_test = y_test.copy()
DS.X_counterfactual = X_counterfactual.copy()
DS.y_counterfactual = y_counterfactual.copy()




d = dice_ml.Data(dataframe=X_train,
                 continuous_features=['age', 'hours_per_week'],
                 outcome_name='income')

# Pre-trained ML model
m = dice_ml.Model(model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
                  backend='TF2', func="ohe-min-max")
# DiCE explanation instance
exp = dice_ml.Dice(d,m)