import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, confusion_matrix, f1_score, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels
import bias_metrics

def _f1_precision_recall(y_true, y_pred):
    metrics = {}
    round_digits = 4
    try:
        metrics['f1'] = round(
            f1_score(y_true, y_pred, zero_division=0), round_digits)
        metrics['precision'] = round(
            precision_score(y_true, y_pred, zero_division=0), round_digits)
        metrics['recall'] = round(recall_score(
            y_true, y_pred, zero_division=0), round_digits)
    except ValueError as ve:
        labels = unique_labels(y_true, y_pred)
        if(len(labels) > 2):
            print(len(labels) + " labels given, expected number 2")
    except Exception as ex:
        print(ex)
    return metrics

def get_binary_classification_metrics(y_true, y_pred, eval_metrics=[]):
    metrics = {}
    round_digits = 4
    metrics = _f1_precision_recall(y_true, y_pred)
    try:
        labels = unique_labels(y_true)
        metrics['average precision'] = round(
            average_precision_score(y_true, y_pred), round_digits)
        if len(labels) > 1:
            metrics['roc auc'] = round(
                roc_auc_score(y_true, y_pred), round_digits)
            metrics['negative log loss'] = round(
                log_loss(y_true, y_pred), round_digits)
    except ValueError as ve:

        labels = unique_labels(y_true, y_pred)
        if(len(labels) > 2):
            print(len(labels) + " labels given, expected number 2")
    except Exception as ex:
        print(ex)

    return metrics

def get_multi_label_classification_metrics(y_true, y_pred, eval_metrics=[]):

    metrics = {}
    round_digits = 4
    multilabel_average_options = ['micro', 'macro', 'weighted']
    for avg in multilabel_average_options:
        try:
            metrics['f1 ' + avg] = round(f1_score(y_true, y_pred, average=avg, zero_division=0), round_digits)
            metrics['precision ' + avg] = round(precision_score(y_true, y_pred, average=avg, zero_division=0), round_digits)
            metrics['recall ' + avg] = round(recall_score(y_true, y_pred, average=avg, zero_division=0), round_digits)
        except Exception as ex:
            raise Exception
           
    return metrics

def get_eval_metrics_for_classificaton(y_true, y_pred, eval_metrics={}):
    """Compute and package different metrics for classification problem,     and return a dictionary with metrics
    :param y_true: 1d array-like, with ground truth (correct target values)
    :type y_true: `array`
    :param y_pred: 1d array-like, estimation returned by classifier.
    :type y_pred: `array`
    :param eval_metrics: metrics requested by the user
    :type eval_metrics: list of metrics which needs to be sent back.
    :return: dictionary with metrics
    :rtype: `dict`
    """
    metrics = {}
    try:
        y_pred = (np.asarray(y_pred)).round()
        y_true = (np.asarray(y_true)).round()
        if y_pred[0].shape != y_true[0].shape:          
            y_pred = list(np.argmax(y_pred, axis=-1))
        labels = unique_labels(y_true, y_pred)
        if len(labels) <= 2:
            metrics = get_binary_classification_metrics(y_true, y_pred, eval_metrics)
        else:
            metrics = get_multi_label_classification_metrics(y_true, y_pred, metrics)
    except Exception as ex:
        raise Exception
    return metrics

def predict( model, x):
    return model.predict(x)
    
def evaluate_model( model, x, y, cols,sensitive, **kwargs):
    acc = {}
    acc['score'] = model.score(x, y, **kwargs)
    y_pred = predict(model, x, **kwargs)
    acc['acc'] = acc['score']
    additional_metrics = get_eval_metrics_for_classificaton(y, y_pred)
    fairness = bias_metrics.fairness_report(x, y, y_pred,sensitive, cols)
    dict_metrics = {**acc, **additional_metrics, **fairness}
    return dict_metrics

def evaluate( model, test_dataset, cols,sensitive):
    if type(test_dataset) is tuple:
        x_test = test_dataset[0]
        y_test = test_dataset[1]
        return evaluate_model(model, x_test, y_test, cols,sensitive)
    else:
        raise Exception
