"""
Bias and Fairness Metrics Service
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger()


class BiasDetector:
    """Fairness and bias detection for ML models"""
    
    def __init__(self, model, sensitive_attr: str):
        """
        Initialize bias detector
        
        Args:
            model: Trained ML model
            sensitive_attr: Name of sensitive attribute column (e.g., 'gender', 'ethnicity')
        """
        self.model = model
        self.sensitive_attr = sensitive_attr
        self.metrics = {}
        self.warnings = []
    
    def compute_fairness_metrics(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        sensitive: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute comprehensive fairness metrics
        
        Args:
            X: Feature DataFrame
            y_true: True labels
            sensitive: Sensitive attribute values
            
        Returns:
            Dictionary with fairness metrics per group
        """
        try:
            logger.info(f"Computing fairness metrics for attribute: {self.sensitive_attr}")
            
            # Get predictions
            y_pred = self.model.predict(X)
            
            # Get unique groups
            groups = sensitive.unique()
            logger.info(f"Found {len(groups)} groups: {list(groups)}")
            
            # Initialize results
            results = {
                "sensitive_attribute": self.sensitive_attr,
                "groups": list(groups),
                "group_metrics": {},
                "overall_metrics": {},
                "warnings": []
            }
            
            # Compute metrics per group
            for group in groups:
                mask = (sensitive == group)
                group_size = mask.sum()
                
                # Check minimum group size
                if group_size < settings.min_group_size:
                    warning = f"Group '{group}' has only {group_size} samples (< {settings.min_group_size}). Metrics may be unreliable."
                    results["warnings"].append(warning)
                    logger.warning(warning)
                
                # Extract group data
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                # Compute group metrics
                group_metrics = self._compute_group_metrics(
                    y_true_group,
                    y_pred_group,
                    group_size
                )
                
                results["group_metrics"][str(group)] = group_metrics
            
            # Compute overall fairness metrics
            results["overall_metrics"] = self._compute_overall_fairness(
                results["group_metrics"]
            )
            
            # Generate warnings for fairness violations
            self._check_fairness_violations(results)
            
            logger.success("Fairness metrics computed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Fairness computation failed: {e}")
            raise RuntimeError(f"Fairness metrics computation failed: {e}")
    
    def _compute_group_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        group_size: int
    ) -> Dict[str, float]:
        """
        Compute metrics for a single group
        
        Args:
            y_true: True labels for group
            y_pred: Predicted labels for group
            group_size: Number of samples in group
            
        Returns:
            Dictionary with group metrics
        """
        # Basic counts
        total = len(y_true)
        positive_pred = (y_pred == 1).sum()
        positive_true = (y_true == 1).sum()
        negative_true = (y_true == 0).sum()
        
        # Confusion matrix components
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        # Compute metrics
        metrics = {
            "group_size": int(group_size),
            "positive_rate": float(positive_pred / total) if total > 0 else 0.0,
            "accuracy": float((tp + tn) / total) if total > 0 else 0.0,
            "true_positive_rate": float(tp / positive_true) if positive_true > 0 else 0.0,
            "true_negative_rate": float(tn / negative_true) if negative_true > 0 else 0.0,
            "false_positive_rate": float(fp / negative_true) if negative_true > 0 else 0.0,
            "false_negative_rate": float(fn / positive_true) if positive_true > 0 else 0.0,
            "precision": float(tp / positive_pred) if positive_pred > 0 else 0.0,
            "recall": float(tp / positive_true) if positive_true > 0 else 0.0
        }
        
        # F1 Score
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = float(
                2 * (metrics["precision"] * metrics["recall"]) / 
                (metrics["precision"] + metrics["recall"])
            )
        else:
            metrics["f1_score"] = 0.0
        
        return metrics
    
    def _compute_overall_fairness(
        self,
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compute overall fairness metrics across groups
        
        Args:
            group_metrics: Metrics per group
            
        Returns:
            Overall fairness metrics
        """
        # Extract metrics
        positive_rates = [m["positive_rate"] for m in group_metrics.values()]
        accuracies = [m["accuracy"] for m in group_metrics.values()]
        tprs = [m["true_positive_rate"] for m in group_metrics.values()]
        
        # Demographic Parity (difference in positive rates)
        demographic_parity_diff = max(positive_rates) - min(positive_rates)
        
        # Equal Opportunity (difference in TPR)
        equal_opportunity_diff = max(tprs) - min(tprs)
        
        # Accuracy Parity (difference in accuracy)
        accuracy_parity_diff = max(accuracies) - min(accuracies)
        
        # Disparate Impact (ratio of min to max positive rate)
        if max(positive_rates) > 0:
            disparate_impact = min(positive_rates) / max(positive_rates)
        else:
            disparate_impact = 1.0
        
        # Statistical Parity (similar to demographic parity)
        statistical_parity = 1.0 - demographic_parity_diff
        
        return {
            "demographic_parity_difference": float(demographic_parity_diff),
            "equal_opportunity_difference": float(equal_opportunity_diff),
            "accuracy_parity_difference": float(accuracy_parity_diff),
            "disparate_impact": float(disparate_impact),
            "statistical_parity": float(statistical_parity),
            "min_positive_rate": float(min(positive_rates)),
            "max_positive_rate": float(max(positive_rates)),
            "min_accuracy": float(min(accuracies)),
            "max_accuracy": float(max(accuracies))
        }
    
    def _check_fairness_violations(self, results: Dict[str, Any]):
        """
        Check for fairness violations and add warnings
        
        Args:
            results: Results dictionary to update with warnings
        """
        overall = results["overall_metrics"]
        
        # Check disparate impact (80% rule)
        if overall["disparate_impact"] < settings.disparate_impact_threshold:
            warning = f"⚠️ Disparate Impact Violation: {overall['disparate_impact']:.3f} < {settings.disparate_impact_threshold} (80% rule)"
            results["warnings"].append(warning)
            logger.warning(warning)
        
        # Check demographic parity
        if overall["demographic_parity_difference"] > settings.demographic_parity_threshold:
            warning = f"⚠️ Demographic Parity Violation: Difference of {overall['demographic_parity_difference']:.3f} > {settings.demographic_parity_threshold}"
            results["warnings"].append(warning)
            logger.warning(warning)
        
        # Check equal opportunity
        if overall["equal_opportunity_difference"] > settings.equal_opportunity_threshold:
            warning = f"⚠️ Equal Opportunity Violation: TPR difference of {overall['equal_opportunity_difference']:.3f} > {settings.equal_opportunity_threshold}"
            results["warnings"].append(warning)
            logger.warning(warning)


def compute_bias_metrics(
    model: Any,
    data: pd.DataFrame,
    feature_names: List[str],
    true_label_col: str,
    sensitive_attr: str
) -> Dict[str, Any]:
    """
    Main function to compute bias metrics
    
    Args:
        model: Trained model
        data: DataFrame with features, true labels, and sensitive attribute
        feature_names: List of feature column names
        true_label_col: Name of true label column
        sensitive_attr: Name of sensitive attribute column
        
    Returns:
        Dictionary with bias metrics
    """
    try:
        logger.info(f"Computing bias metrics for sensitive attribute: {sensitive_attr}")
        
        # Validate columns
        required_cols = feature_names + [true_label_col, sensitive_attr]
        missing_cols = set(required_cols) - set(data.columns)
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Extract data
        X = data[feature_names]
        y_true = data[true_label_col]
        sensitive = data[sensitive_attr]
        
        # Initialize detector
        detector = BiasDetector(model, sensitive_attr)
        
        # Compute metrics
        results = detector.compute_fairness_metrics(X, y_true, sensitive)
        
        return results
        
    except Exception as e:
        logger.error(f"Bias metrics computation failed: {e}")
        raise