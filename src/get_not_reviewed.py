import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_train_data_for_gemma(TRAIN_SET,test_size = None):
    raw_df = pd.read_csv(TRAIN_SET)
    df = raw_df[['PatientID','MRN','AccessionNumber','Narrative', 'nec_features', 'pneumatosis', 'pvg','freeair']]

    if test_size is not None:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['nec_features'], random_state=42)
        return train_df, test_df

print('loading training data.....')


test_size_ls = [0.2,0.4,0.6,0.8]
for test_size in test_size_ls:
    LLM_ready2review_df,LLM_sstrain_df = load_train_data_for_gemma(TRAIN_SET,test_size = test_size)
    print('The ready data for LLM is:',LLM_ready2review_df.shape[0])
    print('The data for training is: ', LLM_sstrain_df.shape[0])
    LLM_ready2review_df.to_csv(f'LLM_ready2review_df_{test_size}.csv')
    LLM_ready2review_df.to_csv(f'LLM_sstrain_df_{test_size}.csv')

