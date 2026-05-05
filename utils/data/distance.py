import numpy as np
from abc import ABC,abstractmethod
import random
import math
import copy
from scipy.stats import entropy
import scipy

class NumericDistance(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def measure(self):
        pass


class DistanceMetric(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def measure(self):
        pass

class StandardDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:,num_feat].std(axis=0, dtype=np.float64)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        distance = X2[:,self.num_feat].copy()
        distance = abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance,axis=1)
        return distance
    def score_list(self,candidates,distances,X,k,change_cost):
        alpha = .5
        betha = .5
        scores = []
        max_cost = np.dot(change_cost,np.transpose(np.ones_like(X)))
        max_possible = distances[np.argmax(distances)] + max_cost
        for candidate,distance in zip(candidates,distances):    
            nonoverlapping = np.where(candidate == X, 0, 1) #[0 if x == y else 1 for x, y in zip(candidate, X)]
            # the cost of changing nonverlapping features divided by the sum of the cost of change of all features
            cost = np.dot(change_cost,np.transpose(nonoverlapping)) / (max_cost + self.data.eps)
            scores.append(max_possible-(alpha*cost + betha*distance))
        return scores

class MinMaxDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:, num_feat].max(axis=0) - X_train[:, num_feat].min(axis=0)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        # if X1.ndim == 1:
        #     X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        distance = X2[:,self.num_feat].copy()
        distance = abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance,axis=1)
        return distance
    def max_distance(self):
        # Calculate the maximum numeric distance using normalized Manhattan distance
        max_num_distance = np.sum((self.scale) / self.scale)
        return max_num_distance


    def score_list(self,candidates,distances,X,k,change_cost):
        alpha = .5
        betha = .5
        scores = []
        max_cost = np.dot(change_cost,np.transpose(np.ones_like(X)))
        max_possible = distances[np.argmax(distances)] + max_cost
        for candidate,distance in zip(candidates,distances):    
            nonoverlapping = np.where(candidate == X, 0, 1) #[0 if x == y else 1 for x, y in zip(candidate, X)]
            # the cost of changing nonverlapping features divided by the sum of the cost of change of all features
            cost = np.dot(change_cost,np.transpose(nonoverlapping)) / (max_cost + self.data.eps)
            scores.append(max_possible-(alpha*cost + betha*distance))
        return scores
    
class HEOM(DistanceMetric):
    def __init__(self, data, numeric_distance:NumericDistance):
        self.data = data
        self.numeric_distance = numeric_distance(data.X_train,data.num_feat,data.eps)
    def measure(self,X1,X2):
        # if X1.ndim == 1:
            # X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        num_distance = self.numeric_distance.measure(X1,X2)
        cat_distance = np.sum(X2[:, self.data.cat_feat] != X1[0, self.data.cat_feat],axis=1)
        distance = num_distance + cat_distance
        return distance
    
    def max_distance(self):
        # Calculate the maximum numeric distance using normalized Manhattan distance
        max_num_distance = np.sum((self.data.X_train[:, self.data.num_feat].max(axis=0) - 
                                   self.data.X_train[:, self.data.num_feat].min(axis=0)) / self.numeric_distance.scale)
        # Maximum categorical distance is the number of categorical features
        max_cat_distance = len(self.data.cat_feat)
        # Total maximum distance
        max_distance = max_num_distance + max_cat_distance
        return max_distance


    def score_list(self,candidates,distances,X,k,change_cost):
        alpha = .5
        betha = .5
        scores = []
        max_cost = np.dot(change_cost,np.transpose(np.ones_like(X)))
        max_possible = distances[np.argmax(distances)] + max_cost
        for candidate,distance in zip(candidates,distances):    
            nonoverlapping = np.where(candidate == X, 0, 1) #[0 if x == y else 1 for x, y in zip(candidate, X)]
            # the cost of changing nonverlapping features divided by the sum of the cost of change of all features
            cost = np.dot(change_cost,np.transpose(nonoverlapping)) / (max_cost + self.data.eps)
            scores.append(max_possible-(alpha*cost + betha*distance))
        return scores
    
class NearestNeighbour:
    def __init__(self,data,distance_metric:DistanceMetric):
        self.data = data
        self.distance_metric = distance_metric

    def find_neighbour(self,X,no_cf = None):
        if self.data.candidates_view.size == 0:
            return no_cf
        distances = self.distance_metric.measure(X,self.data.candidates_view)
        if distances.size == 0:
            #### maybe should be updated to prevent invalid resutls. for now we use it
            return no_cf
            #raise ValueError("Distances array is empty. Check the inputs to the measure method.")
        min_idx = distances.argmin()
        return self.data.candidates_view[min_idx, :].copy()[np.newaxis, :]
    
    def find_neighbour_from_candidates(self,X,candidate_view):
        distances = self.distance_metric.measure(X,candidate_view)
        if distances.size == 0:
            return self.NotFoundCF
            #raise ValueError("Distances array is empty. Check the inputs to the measure method.")
        min_idx = distances.argmin()
        return candidate_view[min_idx, :].copy()[np.newaxis, :]
    

    def find_K_neighbours(self,X,k,random = False):
        candidates = copy.deepcopy(self.data.candidates_view)
        ## Find K nearest neighbours 
        if(random == False):
            distances = self.distance_metric.measure(X,candidates)
            if distances.size == 0:
                raise ValueError("Distances array is empty. Check the inputs to the measure method.")
            K_neighours = []
            for i in range(k):
                min_idx = distances.argmin()
                neighbor = candidates[min_idx, :].copy()[np.newaxis, :] 

                K_neighours.append(neighbor)
                candidates = np.delete(candidates, min_idx, axis=0)
                distances = np.delete(distances, min_idx)
        else:  ## randomly select K instances from target class
            K_neighours = []
            for i in range(k):
                random_idx = np.random.choice(candidates.shape[0])                
                neighbor = candidates[random_idx, :].copy()[np.newaxis, :] 

                K_neighours.append(neighbor)
                candidates = np.delete(candidates, random_idx, axis=0)

        return K_neighours
    

    def check_actionablity(self,a,b):
        for index, value in enumerate(a):
            if value == 1 and index not in b:
                return False
        return True
            
    def find_plause_neighbour(self,X,changable_features=None):  #now both functions can mix
        distances = self.distance_metric.measure(X,self.data.candidates_view)
        #####
        ###sort candidate indexes based on distance, then select the first ine which its changable features are the same as original one
        
        if(changable_features != None):
            indexed_array = list(enumerate(distances))
            sorted_indexes = [index for index, _ in sorted(indexed_array, key=lambda x: x[1])]
            farthest_index = distances.argmax()
            farthers_neighbor = self.data.candidates_view[farthest_index, :].copy()[np.newaxis, :]
            while(len(sorted_indexes) > 0):
                #pop one index, choose its associated datapoint, compare if its changable feature are the same as original, if not, repeat and pop another index
                idx = sorted_indexes.pop(0)
                basic = self.data.candidates_view[idx, :].copy()[np.newaxis, :]
                
                orig_diff = np.where(X != basic)[1]
                if(changable_features != None):
                    diff = [x for x in orig_diff if x not in changable_features]
                
                if len(diff) == 0:
                    return basic

            print("No plausible candidate")
            #for now return farthest instance, we should solve it later
            return farthers_neighbor
        else: 
            min_idx = distances.argmin()
            return self.data.candidates_view[min_idx, :].copy()[np.newaxis, :]

#orig_diff = np.where(CF_candidate != NN)[1]
        #return self.data.candidates_view[min_idx, :].copy()[np.newaxis, :]
    
class PrivateNearestNeighbour:
    def __init__(self,data,distance_metric:DistanceMetric):
        self.data = data
        self.distance_metric = distance_metric

    def find_K_neighbours(self,change_cost,X,k,epsilon = 1,sensitivity = 1):
        candidates = self.data.candidates_view.copy()
        distances = self.distance_metric.measure(X,self.data.candidates_view)
        #generate a list of neighbours and return it
        scores = self.distance_metric.score_list(candidates=candidates,distances = distances,change_cost = change_cost,X = X,k = k)
        
        K_neighours = []
        for i in range(k):
            neighbor = self.Exponential_mechanism(scores,epsilon,sensitivity)
#            if not any(item == neighbor for item in K_neighours):    #neighbor not in K_neighours:
            K_neighours.append(neighbor)
#            else:
#                i = i-1 
        
        #       min_idx = distances.argmin()
        return K_neighours #self.data.candidates_view[min_idx, :].copy()[np.newaxis, :]
   

    def score_list(self,candidates,distances,X,k,change_cost):
        alpha = .5
        betha = .5
        scores = []
        max_cost = np.dot(change_cost,np.transpose(np.ones_like(X)))
        max_possible = distances[np.argmax(distances)] + max_cost
        for candidate,distance in zip(candidates,distances):    
            nonoverlapping = np.where(candidate == X, 0, 1) #[0 if x == y else 1 for x, y in zip(candidate, X)]
            # the cost of changing nonverlapping features divided by the sum of the cost of change of all features
            cost = np.dot(change_cost,np.transpose(nonoverlapping)) / (max_cost + self.data.eps)
            scores.append(max_possible-(alpha*cost + betha*distance))
        return scores
    
    ##### this function was active and working for AAAI submission
    def score_list_working(self,candidates,distances,X,k,change_cost):
        alpha = .5
        betha = .5
        scores = []
        max_cost = np.dot(change_cost,np.transpose(np.ones_like(X)))
        max_possible = distances[np.argmax(distances)] + max_cost
        for candidate,distance in zip(candidates,distances):    
            nonoverlapping = np.where(candidate == X, 0, 1) #[0 if x == y else 1 for x, y in zip(candidate, X)]
            # the cost of changing nonverlapping features divided by the sum of the cost of change of all features
            cost = np.dot(change_cost,np.transpose(nonoverlapping)) / (max_cost + self.data.eps)
            scores.append(max_possible-(alpha*cost + betha*distance))
        return scores



    def Exponential_mechanism(self,scores,epsilon,sensitivity = 1):
        probabilities = []
        for score in scores:
            probabilities.append(math.exp((epsilon * score)/(2 * sensitivity)))
    
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [prob / total_prob for prob in probabilities]
    
        # Select item based on probabilities
        selected_index = random.choices(range(len(scores)), probabilities)[0]
        selected_item = self.data.candidates_view[selected_index]
        
        return selected_item

    ############# working version of the function for AAAI submission
    def Exponential_mechanism_AAAI(self,scores,epsilon):
        probabilities = []
        for score in scores:
            probabilities.append(math.exp(epsilon * score))
    
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [prob / total_prob for prob in probabilities]
    
        # Select item based on probabilities
        selected_index = random.choices(range(len(scores)), probabilities)[0]
        selected_item = self.data.candidates_view[selected_index]
        
        return selected_item



    def Test_exp_mech(self,change_cost,X):
        #print(self.data.candidates_view)
        distances = self.distance_metric.measure(X,self.data.candidates_view)
        #generate a list of neighbours and return it
        scores = self.distance_metric.score_list(candidates=self.data.candidates_view,distances = distances,change_cost = change_cost,X = X,k = 1)
        epsilon = 10
        selected_scores = []
        selection_count = [0] * np.shape(self.data.candidates_view)[0]
        #print(len(selection_count))        
        for i in range(1000):
            neighbor_index,neighbor_score = self.Exponential_mechanism_testing(scores,epsilon)
#            if not any(item == neighbor for item in K_neighours):    #neighbor not in K_neighours:
            selection_count[neighbor_index]  +=1 
            selected_scores.append(neighbor_score[0][0])
#            else:
#                i = i-1 
        
        #       min_idx = distances.argmin()
        return scores,distances,selection_count,selected_scores #self.data.candidates_view[min_idx, :].copy()[np.newaxis, :]
    
    def Exponential_mechanism_testing(self,scores,epsilon):
        probabilities = []
        for score in scores:
            probabilities.append(math.exp(epsilon * score))
    
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [prob / total_prob for prob in probabilities]
    
        ####################check if probabilities are assigned correctly
        #print("highest probabilities is in index {} with value {}".format(np.argmax(probabilities),probabilities[np.argmax(probabilities)]))
        #print("highest score is in index {} with value {}".format(np.argmax(scores),scores[np.argmax(scores)]))
        ############################################################
        # Select item based on probabilities
        selected_index = random.choices(range(len(scores)), weights=probabilities)[0]
        selected_score = scores[selected_index]
        #selected_item = self.data.candidates_view[selected_index]
        
        return selected_index,selected_score
    
    # Working version of the function for AAAI submission
    def Exponential_mechanism_testing_1(self,scores,epsilon):
        probabilities = []
        for score in scores:
            probabilities.append(math.exp(epsilon * score))
    
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [prob / total_prob for prob in probabilities]
    
        ####################check if probabilities are assigned correctly
        #print("highest probabilities is in index {} with value {}".format(np.argmax(probabilities),probabilities[np.argmax(probabilities)]))
        #print("highest score is in index {} with value {}".format(np.argmax(scores),scores[np.argmax(scores)]))
        ############################################################
        # Select item based on probabilities
        selected_index = random.choices(range(len(scores)), weights=probabilities)[0]
        selected_score = scores[selected_index]
        #selected_item = self.data.candidates_view[selected_index]
        
        return selected_index,selected_score 
    
    
class KLDivergence:
    def __init__(self,dataset):
        self.dataset = dataset
    def distance(self,instance1,instance2):
        prob_dist1 = []
        prob_dist2 = []
        for i in range(self.dataset.shape[1]):
            prob_dist1.append(np.mean(self.dataset[:, i] == instance1[0,i]))
            prob_dist2.append(np.mean(self.dataset[:, i] == instance2[0,i]))

        # Compute KL divergence
        
        kl_divergence = sum(prob_dist2[i] * np.log(prob_dist2[i]/prob_dist1[i]+.0000000001) for i in range(len(prob_dist2)))  # np.sum(prob_dist1*np.log(prob_dist1/prob_dist2))
        #entropy(prob_dist2, prob_dist1) # First instance belongs to original distribution, so we calculate distance of the other to it
        print("KL Divergence:", kl_divergence)
        return kl_divergence




def compute_feature_ranges(X, numerical_features):
    """
    Compute the range (max - min) for numerical features.

    Parameters:
    ----------
    X : np.array
        The dataset.
    numerical_features : list
        Indices of numerical features.

    Returns:
    -------
    np.array
        An array of ranges for numerical features.
    """
    feature_ranges = np.zeros(X.shape[1])
    for i in numerical_features:
        feature_ranges[i] = X[:, i].max() - X[:, i].min()
    return feature_ranges




def heom_distance(instance1, instance2, numerical_features, feature_ranges):
    """
    Compute the HEOM distance between two instances.

    Parameters:
    ----------
    instance1 : np.array
        The first instance (e.g., counterfactual).
    instance2 : np.array
        The second instance (e.g., original instance).
    numerical_features : list
        Indices of numerical features.
    feature_ranges : np.array
        Ranges of numerical features (max - min for each feature).

    Returns:
    -------
    float
        The HEOM distance between the two instances.
    """
    distance = 0.0
    num_features = instance1.shape[0]  ## number of features

    # Numerical features: normalized Euclidean distance
    for i in range(num_features):
        if i in numerical_features:
            if feature_ranges[i] == 0:  # Avoid division by zero
                norm_diff = 0
            else:
                norm_diff = (instance1[i] - instance2[i]) / feature_ranges[i]
            distance += norm_diff ** 2
        else:
            # Categorical features: overlap metric
            if instance1[i] != instance2[i]:
                distance += 1  # Add 1 if categorical values differ

    return np.sqrt(distance)


def heom_distance_new(instance1, instance2, numerical_features, feature_ranges):
    """
    Compute the HEOM distance between two instances.

    Parameters:
    ----------
    instance1 : np.array
        The first instance (e.g., counterfactual).
    instance2 : np.array
        The second instance (e.g., original instance).
    numerical_features : list
        Indices of numerical features.
    feature_ranges : np.array
        Ranges of numerical features (max - min for each feature).

    Returns:
    -------
    float
        The HEOM distance between the two instances.
    """
    distance = 0.0
    num_features = instance1.shape[0]  ## number of features
    f_range = 0.0
    idx = 0
    # Numerical features: normalized Euclidean distance
    for i in range(num_features):
        if i in numerical_features:
            f_range = feature_ranges['numerical']['max'][idx] - feature_ranges['numerical']['min'][idx]
            if f_range == 0:  # Avoid division by zero
                norm_diff = 0
            else:
                norm_diff = (instance1[i] - instance2[i]) / f_range
            distance += norm_diff ** 2
            idx += 1
        else:
            # Categorical features: overlap metric
            if instance1[i] != instance2[i]:
                distance += 1  # Add 1 if categorical values differ

    return np.sqrt(distance)