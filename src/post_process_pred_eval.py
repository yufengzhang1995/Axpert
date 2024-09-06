import pandas as pd
import numpy as np
import os,sys
import re
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score,confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa
import warnings
warnings.filterwarnings('ignore')
import csv
import copy



def convert_ans2num(answer):
        mapping = {"Yes": 1, "No": 0, "Uncertain": 2, "None":0}
        return mapping.get(answer, -1) 

def process_answers(text):
    num_list = []
    answer_book = re.findall(r"model <answer>(.*?)</answer>", text, re.DOTALL) # 0.6
    if answer_book:
        answer_list = answer_book[0].strip().split('\n')
        for answer in answer_list:
            answer = answer.split('.')[-1].strip()
            num = convert_ans2num(answer)
            num_list.append(num)
    else:
        num_list = [0,0,0,0]
    return num_list

def compute_metrics(y_true, y_pred, cls = 'pos'):
    if cls == 'pos':
        y_true[y_true == 2] = 0
        y_pred[y_pred == 2] = 0
        f1 = f1_score(y_true, y_pred,  )
        recall = recall_score(y_true, y_pred,  )
        precision = precision_score(y_true, y_pred,zero_division=0, )
        accuracy = accuracy_score(y_true, y_pred)
    elif cls == 'uncertain':
        y_true[y_true == 1] = 0
        y_pred[y_pred == 1] = 0
        f1 = f1_score(y_true, y_pred,pos_label=2)
        recall = recall_score(y_true, y_pred,pos_label=2)
        precision = precision_score(y_true, y_pred,zero_division=0, pos_label=2)
        accuracy = accuracy_score(y_true, y_pred)
    elif cls == 'neg':
        y_true[y_true == 2] = 1
        y_pred[y_pred == 2] = 1
        f1 = f1_score(y_true, y_pred,  pos_label=0)
        recall = recall_score(y_true, y_pred,  pos_label=0)
        precision = precision_score(y_true, y_pred,zero_division=0, pos_label=0)
        accuracy = accuracy_score(y_true, y_pred)
    return f1,recall,precision,accuracy
            
test_size =  [1.0] # 0.2,0.4,0.6,0.8,
for rs in [0,1,2]:
    for ts in test_size:
        file_path = os.path.join(FILE_PATH,f'ts_{ts}_rs_{rs}')
        file_names = os.listdir(file_path)
        metric_path = os.path.join(METRIC_PATH,f'ts_{ts}_rs_{rs}')
        if not os.path.isdir(metric_path):
            os.mkdir(metric_path)
        
        for file in file_names:
            if ('test_imp' in file) and ('csv' in file) and ('epoch20' in file) : # and ('kw' in file)
                save_name = file.split('/')[-1].strip()
                print(f'################ save name {save_name} and metric_path {metric_path} ################')
                FILE = os.path.join(file_path,file)
                df = pd.read_csv(FILE)

                df['pred_numeric'] = df['pred'].apply(process_answers)

                y_pred = []
                y_gt = []
                
                invalid_idx = []
                
                for i in range(df.shape[0]):
                    y_pred.append(df['pred_numeric'].iloc[i])
                    if -1 in df['pred_numeric'].iloc[i]:
                        invalid_idx.append(i)
                    y_gt.append([df['nec_features'].iloc[i],
                                df['pneumatosis'].iloc[i],
                                df['pvg'].iloc[i],
                                df['freeair'].iloc[i]])

                # only for test can be commented
                with open('y_pred.txt', 'w') as file:
                    for item in y_pred:
                        file.write(f"{item}\n")
                        
                with open('y_gt.txt', 'w') as file:
                    for item in y_gt:
                        file.write(f"{item}\n")
                
                for i,s in enumerate(y_pred):
                    if len(s) != 4:
                        invalid_idx.append(i)
                
                
                y_gt= [item for index, item in enumerate(y_gt) if index not in invalid_idx]
                y_pred = [item for index, item in enumerate(y_pred) if index not in invalid_idx]
                
                print('*** NOTE! The invalid index is :',invalid_idx)
                
                
                y_gt = np.array(y_gt)
                y_pred = np.array(y_pred)
                y_gt = np.where(np.isnan(y_gt), 0, y_gt)

                feature_names = ['nec_features','pneumatosis','pvg','freeair']
                results = []
                # print('******* Using model ******:',save_name)
                metrics_path = os.path.join(metric_path,f'feature_metrics_{save_name}.csv')
                with open(metrics_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['Condition', 'F1_neg', 'F1_pos', 'F1_uncertain',
                                                    'Precision_neg', 'Precision_pos','Precision_uncertain',
                                                    'Recall_neg', 'Recall_pos', 'Recall_uncertain','average_Kappa',
                                                    'Accuracy_neg', 'Accuracy_pos', 'Accuracy_uncertain'])
                    for (i,feat) in enumerate(feature_names):
                        f1s = []
                        recalls = []
                        precisions = []
                        accuracys = []
                        for cls in ['neg','pos','uncertain']:
                            f1,recall,precision,accuracy = compute_metrics(copy.deepcopy(y_gt[:,i]), copy.deepcopy(y_pred[:,i]),cls = cls)
                            f1s.append(f1)
                            recalls.append(recall)
                            precisions.append(precision)
                            accuracys.append(accuracy)
                        mat = confusion_matrix(y_gt[:,i], y_pred[:,i])
                        kappa = cohens_kappa(mat, return_results=False)

                        metrics = {
                            'f1':f1s,
                            'precision': precisions,
                            'recall': recalls,
                            'accuracy': accuracys,
                            'kappa': kappa}

                        print(' %s: \n'
                        ' \t f1_neg: %.3f, f1_pos: %.3f,  f1_uncertain: %.3f,\n'
                        '\t precision_neg: %.3f, precision_pos: %.3f, precision_uncertain: %.3f,\n'
                        '\t recall_neg: %.3f, recall_pos: %.3f, recall_uncertain: %.3f, \n '
                        '\t kappa: %.3f, accuracy_neg: %.3f,  accuracy_pos: %.3f, accuracy_uncertain: %.3f,' % (
                                feat,
                                metrics['f1'][0],
                                metrics['f1'][1],
                                metrics['f1'][2],
                                metrics['precision'][0],
                                metrics['precision'][1],
                                metrics['precision'][2],
                                metrics['recall'][0],
                                metrics['recall'][1],
                                metrics['recall'][2],
                                metrics['kappa'],
                                metrics['accuracy'][0],
                                metrics['accuracy'][1],
                                metrics['accuracy'][2]
                        ))
                        print('====================================')
                        csv_writer.writerow([
                                feat,
                                metrics['f1'][0],
                                metrics['f1'][1],
                                metrics['f1'][2],
                                metrics['precision'][0],
                                metrics['precision'][1],
                                metrics['precision'][2],
                                metrics['recall'][0],
                                metrics['recall'][1],
                                metrics['recall'][2],
                                metrics['kappa'],
                                metrics['accuracy'][0],
                                metrics['accuracy'][1],
                                metrics['accuracy'][2],
                            ])
                print(f" ################ Svaing finished ")
            
            
            
