import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import utils
from bert_labeler import bert_labeler
from bert_tokenizer import tokenize
from transformers import BertTokenizer
from collections import OrderedDict
from UnlabeledDataset import UnlabeledDataset
from impressions_dataset import ImpressionsDataset
from constants import *
from tqdm import tqdm
import copy
from run_bert import collate_fn_labels
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score, average_precision_score, roc_auc_score,confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa
import csv
import re

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

def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
        where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
        keys 'imp', 'len' as returned by the __getitem__
        function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
        'imp' is a tensor with padding and batch size as the
        first dimension. 'len' is a list of the length of 
        each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch

def load_unlabeled_data(csv_path,
                        batch_size=1, 
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        test = False,
                        list_path = None):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """

    if test:
        print('Using impression dataset')
        collate_fn = collate_fn_labels
        dset = ImpressionsDataset(csv_path, list_path)
    else:
        print('Using unlabeled dataset')
        collate_fn = collate_fn_no_labels
        dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader
    
def label(checkpoint_path, csv_path, test = False, list_path = None):
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld = load_unlabeled_data(csv_path,test = test,
                        list_path = list_path)
    
    # for batch_idx, batch in enumerate(ld):
    #     print(f"Batch {batch_idx + 1}:")
    #     print(f"Impressions: {batch['imp']}")
    #     print(f"Labels: {batch['label']}")
    #     print(f"Lengths: {batch['len']}")
    #     # Inspecting only the first few batches
    #     if batch_idx == 2:  # Change this number to inspect more or fewer batches
    #         break
    
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: 
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = model.to(device)
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
    model = model.to(device)    
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]
    y_gt = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % 1)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)
            label = data['label'] #(batch_size, 3)
            label = label.permute(1, 0).to(device)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)
                y_gt[j].append(label[j])

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)
            y_gt[j] = torch.cat(y_gt[j], dim=0)
             
    if was_training:
        model.train()

    y_pred = np.array([t.tolist() for t in y_pred]).squeeze() # (3, 438)
    y_gt = np.array([t.tolist() for t in y_gt]).squeeze() # (3, 438)
    # print('pred')
    # print(y_pred.shape)
    # print('gt')
    # print(y_gt.shape)
    
    ## ----- TODO -------## 
    
    
    
    
    metrics = {}
    
    for j in range(len(CONDITIONS)):
        metrics[CONDITIONS[j]] = {}
        f1s = []
        recalls = []
        precisions = []
        accuracys = []
        for cls in ['neg','pos','uncertain']:
            f1,recall,precision,accuracy = compute_metrics(copy.deepcopy(y_gt[j]), copy.deepcopy(y_pred[j]),cls = cls)
            f1s.append(f1)
            recalls.append(recall)
            precisions.append(precision)
            accuracys.append(accuracy)
        
        
        mat = confusion_matrix(y_gt[j], y_pred[j])
        kappa = cohens_kappa(mat, return_results=False)
        metrics[CONDITIONS[j]] = {
                'f1':f1s,
                'precision': precisions,
                'recall': recalls,
                'accuracy': accuracys,
                'kappa': kappa}
    
    with open('y_pred.txt', 'w') as file:
        for item in y_pred:
            file.write(f"{item}\n")
            
    with open('y_gt.txt', 'w') as file:
        for item in y_gt:
            file.write(f"{item}\n")

    for j in range(len(CONDITIONS)):
        print('%s f1_neg: %.3f, f1_pos: %.3f,  f1_uncertain: %.3f,\t'
                'precision_neg: %.3f, precision_pos: %.3f, precision_uncertain: %.3f,\t'
                'recall_neg: %.3f, recall_pos: %.3f, recall_uncertain: %.3f, \t '
                'kappa: %.3f, accuracy_neg: %.3f,  accuracy_pos: %.3f, accuracy_uncertain: %.3f,' % (
                CONDITIONS[j],
                metrics[CONDITIONS[j]]['f1'][0],
                metrics[CONDITIONS[j]]['f1'][1],
                metrics[CONDITIONS[j]]['f1'][2],
                metrics[CONDITIONS[j]]['precision'][0],
                metrics[CONDITIONS[j]]['precision'][1],
                metrics[CONDITIONS[j]]['precision'][2],
                metrics[CONDITIONS[j]]['recall'][0],
                metrics[CONDITIONS[j]]['recall'][1],
                metrics[CONDITIONS[j]]['recall'][2],
                metrics[CONDITIONS[j]]['kappa'],
                metrics[CONDITIONS[j]]['accuracy'][0],
                metrics[CONDITIONS[j]]['accuracy'][1],
                metrics[CONDITIONS[j]]['accuracy'][2]
                ))
    
    metrics_path = os.path.join(out_path, 'test_metrics.csv')
    with open(metrics_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Condition', 'F1_neg', 'F1_pos', 'F1_uncertain',
                                             'Precision_neg', 'Precision_pos','Precision_uncertain',
                                             'Recall_neg', 'Recall_pos', 'Recall_uncertain','average_Kappa',
                                             'Accuracy_neg', 'Accuracy_pos', 'Accuracy_uncertain'])
        
        
        # Write the data rows
        for j in range(len(CONDITIONS)):
            csv_writer.writerow([
                CONDITIONS[j],
                metrics[CONDITIONS[j]]['f1'][0],
                metrics[CONDITIONS[j]]['f1'][1],
                metrics[CONDITIONS[j]]['f1'][2],
                metrics[CONDITIONS[j]]['precision'][0],
                metrics[CONDITIONS[j]]['precision'][1],
                metrics[CONDITIONS[j]]['precision'][2],
                metrics[CONDITIONS[j]]['recall'][0],
                metrics[CONDITIONS[j]]['recall'][1],
                metrics[CONDITIONS[j]]['recall'][2],
                metrics[CONDITIONS[j]]['kappa'],
                metrics[CONDITIONS[j]]['accuracy'][0],
                metrics[CONDITIONS[j]]['accuracy'][1],
                metrics[CONDITIONS[j]]['accuracy'][2],
            ])
    print(f"Training metrics saved to {metrics_path}")

    return y_pred

def save_preds(y_pred, csv_path, out_path):
    """Save predictions as out_path/labeled_reports.csv 
    @param y_pred (List[List[int]]): list of predictions for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    df_pred = pd.DataFrame(y_pred, columns=[f'pred_{cond}' for cond in CONDITIONS])
    df = pd.read_csv(csv_path)
    df_pred = pd.concat([df, df_pred], axis=1)
    
    df_pred.to_csv(os.path.join(out_path,'pred_test.csv'), index=False)
    
    
def extract_epoch_iter(filename):
    match = re.match(r'model_epoch(\d+)_iter(\d+)', filename)
    if match:
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return epoch, iteration
    return None, None

def find_best_model(folder_path):
    best_epoch = -1
    best_iter = -1
    best_model = None

    for file_name in os.listdir(folder_path):
        epoch, iteration = extract_epoch_iter(file_name)
        if epoch is not None and iteration is not None:
            if epoch > best_epoch or (epoch == best_epoch and iteration > best_iter):
                if best_model:  # Delete previous best model as it's now subpar
                    os.remove(os.path.join(folder_path, best_model))
                best_epoch = epoch
                best_iter = iteration
                best_model = file_name
            else:
                print('Delete subpar model {os.path.join(folder_path, file_name}')
                os.remove(os.path.join(folder_path, file_name))

    return best_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    parser.add_argument('-t', '--test', action='store_true', 
                        help='use label information to test or not')
    parser.add_argument('-l', '--imp_list', type=str, default = None,nargs='?', required=False,
                        help='path to json containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-seed', '--seed', type=int, required=True,
                        help='bootstrap seed')
    parser.add_argument('-clinical', action='store_true', help='Toggle bootstrap mode on')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_path
    checkpoint_path = args.checkpoint
    t = args.test
    list_path = args.imp_list
    seed = args.seed
    print('test with label?',t)
    clinical = args.clinical
    if clinical:
        out_path = os.path.join(out_path,'clinical',f'rand_{seed}')
        checkpoint_path = out_path
    else:
        out_path = os.path.join(out_path,'uncased',f'rand_{seed}')
        checkpoint_path = out_path
    print('clinical?:',clinical)
    print(out_path)
    print(checkpoint_path)
    
    # Initialize a dictionary to store the best model for each rand folder
    best_models = {}

    # Loop through each subfolder to find the best model

    # folder_name = f'rand_{seed}'
    folder_path = checkpoint_path
        
    best_model = find_best_model(folder_path)
    print(f'For rand {seed} the best model is: {best_model}')
    checkpoint = os.path.join(folder_path,best_model)
    y_pred = label(checkpoint, csv_path,test = t, list_path = list_path)
    # save_preds(y_pred, csv_path, out_path)