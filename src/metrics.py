import ot
import numpy as np
from scipy.optimize import linear_sum_assignment


# GROUP ASSIGNMENT METRICS =============================

def contingency_matrix(labels_true, labels_pred):
    """
    Build a contingency matrix describing the relationship between labels.
    Implementation to avoid dependency on sklearn.metrics.cluster.contingency_matrix.
    """
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    
    n_classes_true = classes_true.shape[0]
    n_classes_pred = classes_pred.shape[0]
    
    contingency = np.zeros((n_classes_true, n_classes_pred), dtype=np.int64)
    
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            contingency[i, j] = np.sum((labels_true == c_true) & (labels_pred == c_pred))
    
    return contingency

def clustering_accuracy(gt_groups, pred_groups):
    """
    Computes clustering accuracy using the Hungarian algorithm to find the best
    matching between ground truth and predicted clusters.
    Inputs:
    - gt_groups: dict mapping group labels to lists of items (ground truth)
    - pred_groups: dict mapping group labels to lists of items (predicted)
    Returns:
    - accuracy: scalar in [0, 1]
    """
    # map item -> label index
    gt_label_map = {}
    for i, (_, items) in enumerate(gt_groups.items()):
        for x in items:
            gt_label_map[x] = i

    pred_label_map = {}
    for i, (_, items) in enumerate(pred_groups.items()):
        for x in items:
            pred_label_map[x] = i

    items_gt = set(gt_label_map.keys())
    items_pred = set(pred_label_map.keys())

    if items_gt != items_pred:
        raise ValueError("Ground truth and prediction must contain the same items.")

    items = sorted(items_gt)

    y_true = [gt_label_map[x] for x in items]
    y_pred = [pred_label_map[x] for x in items]

    # contingency matrix
    cm = contingency_matrix(y_true, y_pred)

    # Hungarian algorithm (maximize matches)
    row_ind, col_ind = linear_sum_assignment(-cm)

    correct = cm[row_ind, col_ind].sum()
    return correct / len(items)

def clustering_performance(gt_groups, pred_groups) -> dict:
    gt_group_keys = list(gt_groups.keys())
    pred_group_keys = list(pred_groups.keys())

    gt_label_map = {}
    for i, (_, items) in enumerate(gt_groups.items()):
        for x in items:
            gt_label_map[x] = i
    pred_label_map = {}
    for i, (_, items) in enumerate(pred_groups.items()):
        for x in items:
            pred_label_map[x] = i
    items = sorted(gt_label_map.keys())
    y_true = np.asarray([gt_label_map[x] for x in items])
    y_pred = np.asarray([pred_label_map[x] for x in items])
    cm = contingency_matrix(y_true, y_pred)

    purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
    completeness = np.sum(np.max(cm, axis=1)) / np.sum(cm) # TODO: Check equivalence with sklearn.metrics completeness_score

    row_ind, col_ind = linear_sum_assignment(-cm)
    accuracy = cm[row_ind, col_ind].sum() / np.sum(cm)

    # Build mappings
    gt_to_pred_mapping = {
        gt_group_keys[r]: pred_group_keys[c]
        for r, c in zip(row_ind, col_ind)
    }
    pred_to_gt_mapping = {
        pred_group_keys[c]: gt_group_keys[r]
        for r, c in zip(row_ind, col_ind)
    }
    # Find unused groups
    used_pred_groups = set(gt_to_pred_mapping.values())
    unused_pred_groups = [
        g for g in pred_group_keys
        if g not in used_pred_groups
    ]
    used_gt_groups = set(pred_to_gt_mapping.values())
    unused_gt_groups = [
        g for g in gt_group_keys
        if g not in used_gt_groups
    ]
    # Assign unmatched GT groups to unused predicted groups
    for gt_key, pred_key in zip(unused_gt_groups, unused_pred_groups):
        gt_to_pred_mapping[gt_key] = pred_key
        pred_to_gt_mapping[pred_key] = gt_key

    return {
        "purity": purity, # Does each predicted cluster mostly contain items from a single true cluster?
        "completeness": completeness, # Does each true cluster mostly fall into a single predicted cluster?
        "accuracy": accuracy, # Overall clustering accuracy based on optimal matching
        "gt_to_pred_mapping": gt_to_pred_mapping,
        "pred_to_gt_mapping": pred_to_gt_mapping,
        "gt_groups": gt_groups,
        "pred_groups": pred_groups
    }


# LEWIDI METRICS =============================

def cross_entropy(targets, predictions, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
        targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def wasserstein_distance(target, prediction):
    target = np.array(target)
    prediction = np.array(prediction)
    size = len(target)
    optimal_transport_matrix = np.abs(np.arange(size).reshape(-1, 1) - np.arange(size))
    dist_wass = ot.emd2(target, prediction, optimal_transport_matrix)
    return dist_wass

# Function for soft label evaluation for datasets Par and CSC
def average_WS(targets, predictions):
    """
    Calculates the average Wasserstein distance (average WD) between
    corresponding pairs of target and prediction distributions.
    Parameters:
        - targets (list of lists): A list of target distributions.
        - predictions (list of lists): A list of predicted distributions.
    Returns:
        - float: average Wasserstein distance
    """
    distances = [wasserstein_distance(p,t) for p, t in zip(targets, predictions)]
    average_distance = round(sum(distances)/ len(targets), 5)
    return average_distance

def absolute_distance(target, prediction, scale_values):
    """
    Calculates the normalized absolute distance between real and predicted values.
    Parameters:
    	- real (list): A list of actual values.
    	- pred (list): A list of predicted values.
    	- scale_values (float): A scaling factor to normalize the distance.
    Returns:
    	- float: The absolute distance as a percentage of the scale value.
    """
    absolute_distance = 0
    for ann in range(len(target)):
        absolute_distance = absolute_distance + abs(target[ann] - prediction[ann])
    absolute_distance = absolute_distance/len(target)
    return absolute_distance/scale_values #*100

# function for perspectivist evaluation for datasets CSC, Par  (Likert scale)
def mean_absolute_distance(all_real, all_pred, scale_values=5):
    """
    Computes the average normalized absolute distance across multiple samples.
    Parameters:
      - all_real (list of lists): A list containing multiple sets of actual values.
      - all_pred (list of lists): A list containing multiple sets of predicted values.
      - scale_values (float): A scaling factor to normalize the distance.
        corresponds to 10 for Par and to 5 for CSC

    Returns:
      - float: The mean absolute distance across all samples.
    """
    distances = []
    for sample in range(len(all_real)):
        distances.append(absolute_distance(all_real[sample], all_pred[sample], scale_values))

    return float(np.mean(distances))

def average_MD(targets, predictions):
    """
    Calculates the average Manhattan Distance (MD) between corresponding pairs of target and prediction distributions.

    Parameters:
      - targets (list of lists): A list of target distributions.
      - predictions (list of lists): A list of predicted distributions.

    Returns:
      - float: The average Manhattan Distance across all target-prediction pairs.
    """
    distances = []
    for target, prediction in zip(targets, predictions):
        # Compute the Manhattan Distance for a single pair
        distance = sum(abs(p - t) for p, t in zip(prediction, target))
        distances.append(round(distance, 5))

    # Compute and return the average Manhattan Distance
    average_distance = round(sum(distances) / len(distances), 5) if distances else 0
    return average_distance   

def error_rate(targets, predictions):
    """
    Calculates the average error rate between corresponding pairs of target and
    prediction vectors. The match score is defined as 1 minus the proportion of
    correctly matched values (based on absolute error) relative to the number
    of elements in each vector.

    Parameters:
       - targets (list of lists): target vector.
       - predictions (list of lists): predicted vector.

    Returns:
       - float: The average match score across all target-prediction pairs.
    """
    match_scores = []

    for target, prediction in zip(targets, predictions):
        # Compute the total absolute error for the pair
        errors = sum(abs(t - p) for t, p in zip(target, prediction))

        # Compute a normalized match score: higher is better, 1.0 means perfect match
        match_score = round(1- ((len(target) - errors) / len(target)), 5)
        match_scores.append(match_score)

    # Return the average match score across all pairs
    return float(np.mean(match_scores))


# NOTE: Multilabel metrics for VariErrNLI (avoid if VariErrNLI is handled via powerset transformation)

def multilabel_average_MD(targets, predictions):
    """
    Computes the overall soft score by averaging the average Manhattan Distances
    (MD) between predicted and target distributions for each sample.
    For each sample:
    - It uses `average_MD` to compute the average of Manhattan distances between
    corresponding target and predicted distributions within that sample.
    - It returns the mean of these average MDs across all samples.

    Parameters:
        - targets (list of list of lists): Each sample contains a list of target distributions.
        - predictions (list of list of lists): Each sample contains a list of predicted distributions.
    Returns:
        - float: The soft score, rounded to 5 decimal places.
    """
    soft_scores = [average_MD(targets[sample], predictions[sample]) for sample in range(len(targets))]
    return round(sum(soft_scores) / len(soft_scores), 5)

def multilabel_error_rate(all_real, all_pred):
    """
    Calculates the average error rate across multiple multiclass samples.
    Each sample consists of several class vectors (lists), and for each sample,
    an average error rate is computed using the `match_score` function.

    Parameters:
      - all_real (list of list of lists): Ground-truth vectors for all samples.
      - all_pred (list of list of lists): Predicted vectors for all samples.
    Returns:
      - float: The mean of all sample-level match scores.
    """
    multiclass_er = []
    for sample in range(len(all_real)):
        multiclass_er.append(error_rate(all_real[sample], all_pred[sample]))
    return float(np.mean(multiclass_er))

