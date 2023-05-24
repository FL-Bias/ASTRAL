import numpy as np
import pandas as pd 
import cvxpy as cp
import cvxopt
from gensim.matutils import hellinger
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score,     recall_score, accuracy_score


def get_fair_metrics(dataset, pred, sensitive_att, pred_is_dataset=False):
    """
    Measure fairness metrics.    
    Parameters: 
    dataset (pandas dataframe): Dataset
    pred (array): Model predictions
    pred_is_dataset, optional (bool): True if prediction is already part of the dataset, column name 'labels'.
    Returns:
    fair_metrics: Fairness metrics.
    """
    if pred_is_dataset:
        dataset_pred = pred
    else:
        dataset_pred = dataset.copy()
        dataset_pred.labels = pred
        dataset_pred.protected_attribute_names= sensitive_att
        dataset_pred.privileged_protected_attributes='1'
        dataset_pred.unprivileged_protected_attributes='0'
    cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference',  'disparate_impact', 'theil_index']
    obj_fairness = [[0,0,0,1,0]]
    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)
    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 
        classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        acc = classified_metric.accuracy()
        row = pd.DataFrame([[metric_pred.mean_difference(), classified_metric.equal_opportunity_difference(), classified_metric.average_abs_odds_difference(),                    metric_pred.disparate_impact(), classified_metric.theil_index()]], columns  = cols, index = [attr])
        fair_metrics = fair_metrics.append(row)    
        
    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)
    return fair_metrics
def num_pos(y_test):
    """
    Gets number of positive labels in test set.
    :param y_test: Labels of test set
    :type nb_points: `np.array`
    :return: number of positive labels
    :rtype: `int`
    """
    if y_test == []:
        return 0
    else:
        return sum(y_test)


def num_true_pos(y_orig, y_pred):
    """
    Gets number of true positives between test set and model-predicted test set.
    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of true positives
    :rtype: `int`
    """
    a = y_orig == 1
    b = y_pred == 1
    c = np.logical_and(a, b)
    return np.sum(c)


def num_false_pos(y_orig, y_pred):
    """
    Gets number of false positives between test set and model-predicted test set.
    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of false positives
    :rtype: `int`
    """
    a = y_orig == 0
    b = y_pred == 1
    c = np.logical_and(a, b)
    return np.sum(c)


def num_true_neg(y_orig, y_pred):
    """
    Gets number of true negatives between test set and model-predicted test set.
    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of true negatives
    :rtype: `int`
    """
    a = y_orig == 0
    b = y_pred == 0
    c = np.logical_and(a, b)
    return np.sum(c)


def num_false_neg(y_orig, y_pred):
    """
    Gets number of false negatives between test set and model-predicted test set.
    :param y_orig: Labels of test set
    :type y_orig: `np.array`
    :param y_pred: Model-predicted labels of test set
    :type y_pred: `np.array`
    :return: number of false negatives
    :rtype: `int`
    """
    a = y_orig == 1
    b = y_pred == 0
    c = np.logical_and(a, b)
    return np.sum(c)


def tp_rate(TP, pos):
    """
    Gets true positive rate.
    :param TP: Number of true positives
    :type TP: `int`
    :param pos: Number of positive labels
    :type pos: `int`
    :return: true positive rate
    :rtype: `float`
    """
    if pos == 0:
        return 0
    else:
        return TP / pos


def tn_rate(TN, neg):
    """
    Gets true positive rate.
    :param: TP: Number of true negatives
    :type TN: `int`
    :param: pos: Number of negative labels
    :type neg: `int`
    :return: true negative rate
    :rtype: `float`
    """
    if neg == 0:
        return 0
    else:
        return TN / neg


def fp_rate(FP, neg):
    """
    Gets false positive rate.
    :param: FP: Number of false positives
    :type FP: `int`
    :param: neg: Number of negative labels
    :type neg: `int`
    :return: false positive rate
    :rtype: `float`
    """
    if neg == 0:
        return 0
    else:
        return FP / neg


def pp_value(TP, FP):
    """
    Gets positive predictive value, or precision.
    :param: TP: Number of true positives
    :type TP: `int`
    :param: FP:  Number of false positives
    :type FP: `int`
    :return: positive predictive value
    :rtype: `float`
    """
    if TP == 0 and FP == 0:
        return 0
    else:
        return TP / (TP + FP)


def fav_rate(y_pred_group):
    """
    Gets rate of favorable outcome.
    :param y_pred_group: Model-predicted labels of test set for privileged/unprivileged group
    :type y_pred_group: `np.array`
    :return: rate of favorable outcome
    :rtype: `float`
    """
    if y_pred_group == [] or y_pred_group.shape[0] == 0:
        return 0
    else:
        return num_pos(y_pred_group) / y_pred_group.shape[0]


def stat_parity_diff(fav_rate_unpriv, fav_rate_priv):
    """
    Gets statistical parity difference between the unprivileged and privileged groups.
    
    :param fav_rate_unpriv: rate of favorable outcome for unprivileged group
    :type fav_rate_unpriv: `float`
    :param fav_rate_priv: rate of favorable outcome for privileged group
    :type fav_rate_priv: `float`
    :return: statistical parity difference
    :rtype: `float`
    """
    return fav_rate_unpriv - fav_rate_priv


def equal_opp_diff(TPR_unpriv, TPR_priv):
    """
     Gets equal opportunity difference between the unprivileged and privileged groups.
    
    :param: TPR_unpriv: true positive rate for unprivileged group
    :type TPR_unpriv: `float`
    :param: TPR_priv: true positive rate for privileged group
    :type TPR_priv: `float`
    :return: equal opportunity difference
    :rtype: `float`
    """
    return TPR_unpriv - TPR_priv


def discr_idx(f1_unpriv, f1_priv):
    """
     Gets equal opportunity difference between the unprivileged and privileged groups.

    :param: TPR_unpriv: true positive rate for unprivileged group
    :type TPR_unpriv: `float`
    :param: TPR_priv: true positive rate for privileged group
    :type TPR_priv: `float`
    :return: equal opportunity difference
    :rtype: `float`
    """
    return f1_unpriv - f1_priv

def avg_odds(FPR_unpriv, FPR_priv, TPR_unpriv, TPR_priv):
    """
    Gets average odds between the unprivileged and privileged groups.
    
    :param: FPR_unpriv: false positive rate for unprivileged group
    :type FPR_unpriv: `float`
    :param: FPR_priv: false positive rate for privileged group
    :type FPR_priv: `float`
    :param: TPR_unpriv: true positive rate for unprivileged group
    :type TPR_unpriv: `float`
    :param: TPR_priv: true positive rate for privileged group
    :type TPR_priv: `float`
    :return: average odds
    :rtype: `float`
    """
    return ((FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)) / 2


def disparate_impact(fav_rate_unpriv, fav_rate_priv):
    """
    Gets disparate impact between the unprivileged and privileged groups.
    
    :param fav_rate_unpriv: rate of favorable outcome for unprivileged group
    :type fav_rate_priv: `float`
    :param fav_rate_priv: rate of favorable outcome for privileged group
    :type fav_rate_unpriv: `float`
    :return: disparate impact
    :rtype: `float`
    """
    if fav_rate_priv == 0:
        return 0
    else:
        return fav_rate_unpriv / fav_rate_priv

def decision_boundary_covariance( in_X, in_Y, in_Y_pred, in_S, in_Theta):
    n_x = len(in_X)
    n = len(in_Theta)
    X = cp.Parameter((n_x, n))
    Y = cp.Parameter((n_x, 1))
    y = cp.Parameter((n_x, 1))
    sum_x = cp.Parameter((n,1))
    Theta = cp.Parameter((n,1))
    Theta_t = cp.Parameter((1,n))
    x = in_X
    x_np = []
    s_bar = in_S.mean()
    s_np = in_S - s_bar
    for i in range(len(x)):
        x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
        x_np.append(x_tmp * s_np[i])
    x_np = np.array(x_np)
    X_ = []
    for i in range(len(x)):
        x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
        X_.append(x_tmp)
    X.value = np.array(X_)
    sum_x.value = x_np.sum(axis=0).transpose().reshape(n, 1)
    Theta.value = in_Theta.reshape(n,1)
    Theta_t.value = in_Theta.reshape(n,1).transpose()
    Y.value = in_Y.reshape(n_x, 1)
    y.value = in_Y_pred.reshape(n_x, 1)
    return 1/n_x * (Theta_t @ sum_x).value.reshape(-1)[0]

def priv_unpriv_sets(training_data, y_test, y_pred, sensitive_attribute, cols):
    """
    Splits y_test and y_pred into two arrays each, depending on whether the sample associated
    with the label was privileged or unprivileged, with respect to the sensitive attribute.
    
    :param training_data: Feature set
    :type training_data: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y_pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column names
    :type cols: `list`
    :return: privileged and unprivileged y label groups
    :rtype: `float`
    """
    training_data = pd.DataFrame(data=training_data)
    training_data.columns = cols

    p_set = training_data.loc[training_data[sensitive_attribute] >= 0.9]
    unp_set = training_data.loc[training_data[sensitive_attribute] < 0.5]

    a = p_set.index.tolist()
    b = unp_set.index.tolist()

    return y_test[a], y_test[b], y_pred[a], y_pred[b]


def priv_unpriv_sets2(training_data, y_test, y_pred, sensitive_attribute, cols):
    """
    Splits y_test and y_pred into two arrays each, depending on whether the sample associated
    with the label was privileged or unprivileged, with respect to the sensitive attribute.

    :param training_data: Feature set
    :type training_data: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y_pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column names
    :type cols: `list`
    :return: privileged and unprivileged y label groups
    :rtype: `float`
    """
    training_data = pd.DataFrame(data=training_data)
    training_data.columns = cols

    p_set = training_data.loc[training_data[sensitive_attribute] == 1]
    unp_set = training_data.loc[training_data[sensitive_attribute] == 0]

    a = p_set.index.tolist()
    b = unp_set.index.tolist()

    return y_test[a], y_test[b], y_pred[a], y_pred[b]


def get_fairness_metrics(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols):
    """
    Calculates middle terms for fairness metrics.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_test_pred: Test set predictions
    :type y_test_pred: `np.array`
    :param SENSITIVE_ATTRIBUTE:
    :type SENSITIVE_ATTRIBUTE: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: fairness metric variables
    :rtype: `float`
    """
    y_test_priv_set, y_test_unpriv_set, y_pred_priv_set, y_pred_unpriv_set =     priv_unpriv_sets(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols)

    pos_unpriv_set = num_pos(y_test_unpriv_set)
    neg_unpriv_set = y_test_unpriv_set.shape[0] - pos_unpriv_set
    pos_priv_set = num_pos(y_test_priv_set)
    neg_priv_set = y_test_priv_set.shape[0] - pos_priv_set

    TP_unpriv = num_true_pos(y_test_unpriv_set, y_pred_unpriv_set)
    TP_priv = num_true_pos(y_test_priv_set, y_pred_priv_set)
    FP_unpriv = num_false_pos(y_test_unpriv_set, y_pred_unpriv_set)
    FP_priv = num_false_pos(y_test_priv_set, y_pred_priv_set)

    TPR_unpriv = tp_rate(TP_unpriv, pos_unpriv_set)
    TPR_priv = tp_rate(TP_priv, pos_priv_set)
    FPR_unpriv = fp_rate(FP_unpriv, neg_unpriv_set)
    FPR_priv = fp_rate(FP_priv, neg_priv_set)

    fav_rate_unpriv = fav_rate(y_pred_unpriv_set)
    fav_rate_priv = fav_rate(y_pred_priv_set)

    f1_unpriv = f1_score(y_test_unpriv_set, y_pred_unpriv_set)
    f1_priv = f1_score(y_test_priv_set, y_pred_priv_set)


    accuracy_unpriv = accuracy_score(y_test_unpriv_set, y_pred_unpriv_set)
    accuracy_priv = accuracy_score(y_test_priv_set, y_pred_priv_set)

    print('accuracy old {} accuracy recent {}'.format(accuracy_unpriv, accuracy_priv))
    print('True po old {} true po recent {}'.format(TPR_unpriv, TPR_priv))

    return fav_rate_unpriv, fav_rate_priv, TPR_unpriv, TPR_priv, FPR_unpriv, FPR_priv, f1_unpriv, f1_priv

def get_spd_metrics(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols):
    """
    Calculates middle terms for fairness metrics.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_test_pred: Test set predictions
    :type y_test_pred: `np.array`
    :param SENSITIVE_ATTRIBUTE:
    :type SENSITIVE_ATTRIBUTE: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: fairness metric variables
    :rtype: `float`
    """
    _, _, y_pred_priv_set, y_pred_unpriv_set =     priv_unpriv_sets(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols)

    fav_rate_unpriv = fav_rate(y_pred_unpriv_set)
    fav_rate_priv = fav_rate(y_pred_priv_set)

    return fav_rate_unpriv, fav_rate_priv
    
def get_eod_metrics(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols):
    """
    Calculates middle terms for fairness metrics.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_test_pred: Test set predictions
    :type y_test_pred: `np.array`
    :param SENSITIVE_ATTRIBUTE:
    :type SENSITIVE_ATTRIBUTE: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: fairness metric variables
    :rtype: `float`
    """
    y_test_priv_set, y_test_unpriv_set, y_pred_priv_set, y_pred_unpriv_set =     priv_unpriv_sets(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols)

    pos_unpriv_set = num_pos(y_test_unpriv_set)
    pos_priv_set = num_pos(y_test_priv_set)

    TP_unpriv = num_true_pos(y_test_unpriv_set, y_pred_unpriv_set)
    TP_priv = num_true_pos(y_test_priv_set, y_pred_priv_set)

    TPR_unpriv = tp_rate(TP_unpriv, pos_unpriv_set)
    TPR_priv = tp_rate(TP_priv, pos_priv_set)


    return TPR_unpriv, TPR_priv


def get_discr_idx_metrics(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols):
    """
    Calculates middle terms for fairness metrics.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_test_pred: Test set predictions
    :type y_test_pred: `np.array`
    :param SENSITIVE_ATTRIBUTE:
    :type SENSITIVE_ATTRIBUTE: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: fairness metric variables
    :rtype: `float`
    """
    y_test_priv_set, y_test_unpriv_set, y_pred_priv_set, y_pred_unpriv_set = priv_unpriv_sets(x_test, y_test, y_test_pred, SENSITIVE_ATTRIBUTE, cols)
    f1_priv = f1_score(y_test_priv_set, y_pred_priv_set)
    f1_unpriv = f1_score(y_test_unpriv_set, y_pred_unpriv_set)

    return f1_unpriv, f1_priv


def fairness_report(x_test, y_test, y_pred, sensitive_attribute, model_params, cols):
    """
    Gets fairness report, with F1 score, statistical parity difference, equal opportunity odds,
    average odds difference and disparate impact.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y__pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: report
    :rtype: `dict`
    """
    if (isinstance(sensitive_attribute,list)):
        list_fav_rate_unpriv, list_fav_rate_priv = [], []
        spd, eod, ao, di, dbc, discri_idx, list_f1_unpriv, list_f1_priv = [], [], [], [], [], [], [], []
        for sa in sensitive_attribute:
            fav_rate_unpriv, fav_rate_priv, TPR_unpriv, TPR_priv, FPR_unpriv, FPR_priv, f1_unpriv, f1_priv = get_fairness_metrics(
                x_test, y_test, y_pred, sa, cols)
            list_fav_rate_unpriv.append(fav_rate_unpriv)
            list_fav_rate_priv.append(fav_rate_priv)
            
            list_f1_unpriv.append(f1_unpriv)
            list_f1_priv.append(f1_priv)

            spd.append(stat_parity_diff(fav_rate_unpriv, fav_rate_priv))
            eod.append(equal_opp_diff(TPR_unpriv, TPR_priv))
            ao.append(avg_odds(FPR_unpriv, FPR_priv, TPR_unpriv, TPR_priv))
            di.append(disparate_impact(fav_rate_unpriv, fav_rate_priv))
            discri_idx.append(discr_idx(f1_unpriv, f1_priv))
            
            training_data = pd.DataFrame(data=x_test)
            training_data.columns = cols
            s = training_data[sa].to_numpy()
            dbc.append(decision_boundary_covariance(x_test, y_test, y_pred, s, model_params))
        f1 = f1_score(y_test, y_pred)
        f2 = classification_report(y_test, y_pred, output_dict=True)
        report = {'F1_report': f1, 'Classification report': f2, 'Statistical Parity Difference': spd, 'Equal Opportunity Difference': eod, 'Average Odds Difference': ao, 'Disparate Impact': di, 'fav_rate_unpriv': list_fav_rate_unpriv, 'fav_rate_priv': list_fav_rate_priv, 'Decision Boundary Covariance': dbc,
                  'F1_unpriv':list_f1_unpriv, 'F1_priv':list_f1_priv, 'Discrimination Index': discri_idx}
        return report
    else:
        fav_rate_unpriv, fav_rate_priv, TPR_unpriv, TPR_priv, FPR_unpriv, FPR_priv, f1_unpriv, f1_priv = get_fairness_metrics(
            x_test, y_test, y_pred, sensitive_attribute, cols)
        f1 = f1_score(y_test, y_pred)
        spd = stat_parity_diff(fav_rate_unpriv, fav_rate_priv)
        eod = equal_opp_diff(TPR_unpriv, TPR_priv)
        ao = avg_odds(FPR_unpriv, FPR_priv, TPR_unpriv, TPR_priv)
        di = disparate_impact(fav_rate_unpriv, fav_rate_priv)
        discri_idx = discr_idx(f1_unpriv, f1_priv)

        f2=classification_report(y_test,y_pred,output_dict=True)
        
        training_data = pd.DataFrame(data=x_test)
        training_data.columns = cols
        s = training_data[sensitive_attribute].to_numpy()
        dbc = decision_boundary_covariance( x_test, y_test, y_pred, s, model_params)

        report = {'F1_report':f1, 'Classification report':f2 ,'Statistical Parity Difference': spd, 'Equal Opportunity Difference': eod, 'Average Odds Difference': ao, 'Disparate Impact': di,'fav_rate_unpriv':fav_rate_unpriv,'fav_rate_priv':fav_rate_priv, 'Decision Boundary Covariance':dbc,
                  'F1_unpriv':f1_unpriv, 'F1_priv':f1_priv, 'Discrimination Index': discri_idx}

                  
        return report

def spd_report(x_test, y_test, y_pred, sensitive_attribute, cols):
    """
    Gets fairness report, with F1 score, statistical parity difference, equal opportunity odds,
    average odds difference and disparate impact.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y__pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: report
    :rtype: `dict`
    """
    if (isinstance(sensitive_attribute,list)):
        list_fav_rate_unpriv, list_fav_rate_priv = [], []
        spd = []
        for sa in sensitive_attribute:
            fav_rate_unpriv, fav_rate_priv = get_spd_metrics(
                x_test, y_test, y_pred, sa, cols)
            list_fav_rate_unpriv.append(fav_rate_unpriv)
            list_fav_rate_priv.append(fav_rate_priv)
            spd.append(stat_parity_diff(fav_rate_unpriv, fav_rate_priv))         
        return {'Statistical Parity Difference': spd}
    else:
        fav_rate_unpriv, fav_rate_priv = get_spd_metrics(
            x_test, y_test, y_pred, sensitive_attribute, cols)
        spd = stat_parity_diff(fav_rate_unpriv, fav_rate_priv)
        return {'Statistical Parity Difference': spd}
        
def eod_report(x_test, y_test, y_pred, sensitive_attribute, cols):
    """
    Gets fairness report, with F1 score, statistical parity difference, equal opportunity odds,
    average odds difference and disparate impact.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y__pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: report
    :rtype: `dict`
    """
    
    if (isinstance(sensitive_attribute,list)):
        eod = []
        for sa in sensitive_attribute:
            TPR_unpriv, TPR_priv = get_eod_metrics(x_test, y_test, y_pred, sa, cols)
            eod.append(equal_opp_diff(TPR_unpriv, TPR_priv))     
        return {'Equal Opportunity Difference': eod}
        
    else:
        TPR_unpriv, TPR_priv = get_eod_metrics( x_test, y_test, y_pred, sensitive_attribute, cols)
        eod = equal_opp_diff(TPR_unpriv, TPR_priv)
        return {'Equal Opportunity Difference': eod}


def discr_idx_report(x_test, y_test, y_pred, sensitive_attribute, cols):
    """
    Gets fairness report, with F1 score, statistical parity difference, equal opportunity odds,
    average odds difference and disparate impact.

    :param x_test: Test feature set
    :type x_test: `np.array`
    :param y_test: Test set labels
    :type y_test: `np.array`
    :param y_pred: Test set predictions
    :type y__pred: `np.array`
    :param sensitive_attribute:
    :type sensitive_attribute: `str`
    :param cols: Feature set column namess
    :type cols: `list`
    :return: report
    :rtype: `dict`
    """
    
    if (isinstance(sensitive_attribute, list)):
        discri = []
        for sa in sensitive_attribute:
            f1_unpriv, f1_priv = get_discr_idx_metrics(x_test, y_test, y_pred, sa, cols)

            discri.append(discr_idx(f1_unpriv, f1_priv))
        return {'Discrimination Index': discri}

    else:
        f1_unpriv, f1_priv = get_discr_idx_metrics(x_test, y_test, y_pred, sensitive_attribute, cols)
        discri = discr_idx(f1_unpriv, f1_priv)
        return {'Discrimination Index': discri}


def uei(y_train, y_train_pred):
    """
    Gets UEI index between training set labels and training set predictions.

    :param y_train:
    :type y_train: `np.array`
    :param y_train_pred:
    :type y_train_pred: `np.array`
    :return: UEI index
    :rtype: `float`
    """
    y_train_norm = y_train / np.sum(y_train)
    if np.sum(y_train_pred) > 0:
        y_train_pred_norm = y_train_pred / np.sum(y_train_pred)
    else:
        y_train_pred_norm = y_train_pred

    return hellinger(y_train_norm, y_train_pred_norm)

