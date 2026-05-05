import numpy as np
import pandas as pd

from typing import Dict, Any, List, Callable, Union, Tuple, Optional

from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)





def reidentify(instance,dataset):
    count = 0
    if(isinstance(dataset, pd.DataFrame)):
        dataset = dataset.to_numpy()
    for i in range(len(dataset)):
        a = np.where(instance == dataset[i])[1]
        if len(a) == len(instance[0]) :
            count +=1

    return count

def compare_solutions(Factual_instance,basic,CF):   #find how the CF changes in comparison with basic_cf and original instance
        eps = 0.00000000001
        # differencses between Factual_instance and basic Counterfacual
        nonoverlapping_features = np.where(Factual_instance != basic)[1]
        nonoverlapping = len(nonoverlapping_features)   # np.count_nonzero(nonoverlapping_features)
        #difference between Factual_instance and generated counterfactual
        Changes_features = np.where(Factual_instance != CF)[1]
        Changed = len(Changes_features) # np.count_nonzero(Changes_features)
        # ratio of features selected from basic_CF   : in changed feutures, find what is equal to basic
        
        Chosen_from_basic_features = np.where(np.logical_and(Factual_instance != CF , CF == basic))[1]
        Chosen_from_basic = len(Chosen_from_basic_features) # np.count_nonzero(Chosen_from_basic_features)
        # ratio of changes features
        changed_ratio = Changed / len(Factual_instance[0]) # ( nonoverlapping + eps )  #to avoid devision by zero
        # ration of selected from basic instance
        from_basic_ratio = Chosen_from_basic / ( Changed + eps)   #to avoid devision by zero
        return changed_ratio,from_basic_ratio



class AttackEvaluator:
    """A comprehensive model evaluation toolkit with modular metrics and advanced privacy analysis.
    
    This class provides a unified interface for calculating various model performance 
    metrics, including classification metrics, ROC analysis, and privacy-focused evaluations.
    """
    def __init__(self, true_labels: np.ndarray, predicted_scores: np.ndarray):
        """Initialize the AttackEvaluator with ground truth labels and predicted scores.
        
        Args:
            true_labels: Ground truth binary labels.
            predicted_scores: Model's predicted probabilities or scores.
        """
        self.true_labels = np.asarray(true_labels)
        self.predicted_scores = np.asarray(predicted_scores)
        
    def _get_predicted_labels(self, decision_threshold: float = None) -> np.ndarray:
        """Convert predicted scores to binary labels using a threshold.
        
        Args:
            decision_threshold: Classification threshold. Defaults to median of predicted scores.
        
        Returns:
            np.ndarray: Binary predicted labels.
        """
        if decision_threshold is None:
            decision_threshold = np.median(self.predicted_scores)
        return (self.predicted_scores > decision_threshold).astype(int)
    
    def roc_metrics(self, target_fprs: List[float] = None) -> Dict[str, float]:
        """Calculate ROC-related metrics including AUC and TPR at specific FPRs.
        
        Args:
            target_fprs: False Positive Rate targets. Defaults to [0, 0.001, 0.01, 0.1].
        
        Returns:
            dict: ROC metrics including AUC and TPR at specified FPRs.
        """
        # Default FPR targets if not provided
        if target_fprs is None:
            target_fprs = [0, 0.001, 0.01, 0.1]
        
        # Calculate AUC
        metrics = {
            'auc_roc': roc_auc_score(self.true_labels, self.predicted_scores)
        }
        
        # Calculate TPR at specific FPRs
        fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_scores)
        for target_fpr in target_fprs:
            metrics[f'tpr_at_fpr_{target_fpr}'] = np.interp(target_fpr, fpr, tpr)
        
        return metrics
    


class BaseAttacker:
    """Base class for implementing a general attack model.

    This class serves as a base for attack models used in testing the privacy and security 
    of machine learning models. It provides methods for setting hyperparameters, executing attacks, 
    evaluating results, validating input data, and building consistent test data.
    """

    def __init__(self, hyper_parameters: Dict[str, Any] = {}):
        """Initialize the attack model with hyperparameters.
        
        Args:
            hyper_parameters: A dictionary containing the hyperparameters for the attack model.
        """
        self.hyper_parameters = hyper_parameters

    def attack(self, mem: np.ndarray, non_mem: np.ndarray, 
              synth: np.ndarray, ref: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute the attack given member, non-member, synthetic data, and optional reference data.

        Args:
            mem: Member data.
            non_mem: Non-member data.
            synth: Synthetic data.
            ref: Reference data. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Predicted scores
                - np.ndarray: True labels
        """
        self._validate_input_data(mem, non_mem, synth, ref)
        
        X_test = self._build_X_test(mem, non_mem)
        y_true = self._build_y_test(mem, non_mem)

        # Implement the attack logic here
        scores = self._compute_attack_scores(X_test, synth, ref)
        ##### should as some graph drawing to see the reality of the scores
        #scores = np.where(np.isposinf(scores), 1e200, np.where(np.isneginf(scores), -1e200, scores))
        return y_true, scores

    def _compute_attack_scores(self, X_test: np.ndarray, 
                             synth: np.ndarray, 
                             ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the attack scores. This method should be implemented by subclasses.

        Args:
            X_test: Test data (member and non-member).
            synth: Synthetic data.
            ref: Reference data. Defaults to None.

        Returns:
            np.ndarray: Predicted scores.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def eval(self, 
             true_labels: np.ndarray, 
             predicted_scores: np.ndarray, 
             metrics: List[str] = ["roc"], 
             **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the attack using the ModelEvaluator class.
        
        Args:
            predicted_scores: Predicted scores from the attack.
            true_labels: True labels (0 for non-member, 1 for member).
            metrics: List of metrics to compute. Can include "roc", "classification", 
                "privacy", "epsilon". Defaults to ["roc"].
            **kwargs: Additional arguments for metrics computation:
                - target_fprs: List of target false positive rates for ROC metrics
                - decision_threshold: Threshold for classification/privacy metrics
                - confidence_level: Confidence level for epsilon evaluation
                - threshold_method: Method for threshold selection in epsilon evaluation
                - validation_split: Validation split ratio for epsilon evaluation
            
        Returns:
            dict: A dictionary of evaluation results for each metric.
        """
        evaluator = AttackEvaluator(true_labels, predicted_scores)
        results = {}
        
        if "roc" in metrics:
            target_fprs = kwargs.get("target_fprs", [0, 0.001, 0.01, 0.1])
            results.update(evaluator.roc_metrics(target_fprs=target_fprs))
        
        if "classification" in metrics:
            decision_threshold = kwargs.get("decision_threshold", None)
            results.update(evaluator.classification_metrics(decision_threshold=decision_threshold))
        
        if "privacy" in metrics:
            decision_threshold = kwargs.get("decision_threshold", None)
            results.update(evaluator.privacy_metrics(decision_threshold=decision_threshold))
        
        if "epsilon" in metrics:
            confidence_level = kwargs.get("confidence_level", 0.9)
            threshold_method = kwargs.get("threshold_method", "ratio")
            validation_split = kwargs.get("validation_split", 0.1)
            results.update(evaluator.epsilon_evaluator(
            confidence_level=confidence_level, 
            threshold_method=threshold_method, 
            validation_split=validation_split)
        )
        return results

    def get_properties(self) -> Dict[str, Any]:
        """Return the hyperparameters of the model.
        
        Returns:
            dict: The hyperparameters of the model.
        """
        return self.hyper_parameters

    def _validate_input_data(self, mem: np.ndarray, 
                           non_mem: np.ndarray, 
                           synth: np.ndarray, 
                           ref: Optional[np.ndarray] = None
    ) -> None:
        """Validate the input data to ensure consistency across numpy arrays.
        
        Args:
            mem: Member data.
            non_mem: Non-member data.
            synth: Synthetic data.
            ref: Reference data. Defaults to None.

        Raises:
            ValueError: If input arrays have inconsistent shapes or data types.
        """
        data_arrays = [mem, non_mem, synth]
        if ref is not None:
            data_arrays.append(ref)

        # Validate that all arrays have the same column count
        mem_columns = mem.shape[1]
        for array in data_arrays:
            if array.shape[1] != mem_columns:
                raise ValueError("All input arrays must have the same number of columns")

        # Validate that all arrays have the same data type
        mem_dtype = mem.dtype
        for array in data_arrays:
            if array.dtype != mem_dtype:
                raise ValueError("All input arrays must have the same data type")

    def _build_X_test(self, mem: np.ndarray, non_mem: np.ndarray) -> np.ndarray:
        """Build the X_test data by concatenating member and non-member data.
        
        Args:
            mem: Member data.
            non_mem: Non-member data.
        
        Returns:
            np.ndarray: The concatenated X_test data.
        """
        return np.concatenate([mem, non_mem], axis=0)
   
    def _build_y_test(self, mem: np.ndarray, non_mem: np.ndarray) -> np.ndarray:
        """Build the y_test data with labels for member and non-member data.
        
        Args:
            mem: Member data.
            non_mem: Non-member data.
        
        Returns:
            np.ndarray: The y_test labels (1 for members, 0 for non-members).
        """
        return np.concatenate([np.ones(len(mem)),np.zeros(len(non_mem))])