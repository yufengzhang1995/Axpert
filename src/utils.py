import pandas as pd
import numpy as np
import os,sys
import re
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score,confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa

def compute_metrics(y_true, y_pred):
    f1 = []
    recall = []
    precision = []
    accuracy = []
    for j in range(len(y_true)): 
        f1.append(f1_score(y_true[j], y_pred[j], average=None))
        recall.append(recall_score(y_true[j], y_pred[j], average=None))
        precision.append(precision_score(y_true[j], y_pred[j],zero_division=0,average=None))
        accuracy.append(accuracy_score(y_true[j], y_pred[j]))
    return f1,recall,precision,accuracy