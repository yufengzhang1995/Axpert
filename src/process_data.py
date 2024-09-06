import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")


def pre_process_narrative(df):
    imp = df['Narrative']
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True).replace('\s+', ' ', regex=True).str.strip()
    df['Narrative'] = imp
    return df

def generate_input(input):
    prompt = f"""Q: This is a pediatric radiology report: {input}"""
    prompt += '(1) Does the child have necrotizing enterocolitis? A. Yes; B. No. '
    prompt += '(2) Does the child have pneumatosis? A. Yes; B. No. '
    prompt += '(3) Does the child have portal venous gas? A. Yes; B. No. '
    prompt += '(4) Does the child have free air? A. Yes; B. No. '
    return prompt


def generate_instruction():
    return 'Please answer the four numbered questions. Write answer using A/B in between <answer></answer>.'

    
def get_ans(x):
    if x == 1:
        return 'A. Yes'
    else:
        return 'B. No'
def generate_output(jk_nec_features,jk_pneumatosis,jk_pvg,jk_freeair):
    temp = np.array([jk_nec_features,jk_pneumatosis,jk_pvg,jk_freeair])
    results = [get_ans(i) for i in temp]     
    answer = '<answer>\n'
    for result in results:
        answer += f' {result}\n'  
    answer += '</answer>'
    return answer

def generate_prompt_for_gemma(data_point,mode):
    """Gen input text based on a 
        prompt, 
        task instruction, 
        (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenized prompt
    """

    # Generate prompt
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
    
    # Samples with additional context into.
    if mode == 'train' or mode == 'eval':
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    # Without
    elif mode == 'test':
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n"""
    return text





def load_train_data_for_gemma(TRAIN_SET,EOS_TOKEN,mode):
    raw_df = pd.read_csv(TRAIN_SET)
    df = raw_df[['PatientID','MRN','Narrative', 'jk_nec_features', 'jk_pneumatosis', 'jk_pvg','jk_freeair', 'notes']]
    df = pre_process_narrative(df)
    df['input'] =  df['Narrative'].apply(lambda x: generate_input(x))  
    df['instruction'] =  generate_instruction()
    df['output'] = df.apply(lambda row: generate_output(row['jk_nec_features'], row['jk_pneumatosis'], row['jk_pvg'], row['jk_freeair']), axis=1)
    results = df[['PatientID','MRN','input','instruction','output','jk_nec_features', 'jk_pneumatosis', 'jk_pvg','jk_freeair']]
    results['prompt'] = results.apply(lambda row: generate_prompt_for_gemma(row,mode), axis=1)
    results['prompt'] = results['prompt'].apply(lambda x: x + EOS_TOKEN)
    return results

def load_test_data_for_gemma(TEST_SET,EOS_TOKEN):
    raw_df = pd.read_csv(TEST_SET)
    df = raw_df[['PatientID','MRN','Narrative']]
    df = pre_process_narrative(df)
    df['input'] =  df['Narrative'].apply(lambda x: generate_input(x))  
    df['instruction'] =  generate_instruction()
    results = df[['PatientID','MRN','input','instruction']]
    results['prompt'] = results.apply(lambda row: generate_prompt_for_gemma(row,mode = 'test'), axis=1)
    results['prompt'] = results['prompt'].apply(lambda x: x + EOS_TOKEN)
    return results



######################### llama3# ######################

def generate_input_for_llama3(input):
    prompt = f"""Q: This is a pediatric radiology report: {input}"""
    prompt += '(1) Does the child have necrotizing enterocolitis? A. Yes; B. No.'
    prompt += '(2) Does the child have pneumatosis? A. Yes; B. No. '
    prompt += '(3) Does the child have portal venous gas? A. Yes; B. No. '
    prompt += '(4) Does the child have free air? A. Yes; B. No. '
    return prompt


def generate_prompt_for_llama3(data_point,mode):
    # Generate prompt
    prefix_text = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI abdominal radiology report labeler<|eot_id|>.\n<|start_header_id|>user<|end_header_id|>'
    
    instruction = 'Can you answer the four numbered questions based on the report input? Write answer using A/B in between <answer></answer>.<|eot_id|>\n'

    
    # Define the prompt content based on the mode
    if mode == 'train' or mode == 'eval':
        text = f"{prefix_text}Here are the inputs: {data_point['input']}.{instruction}<|start_header_id|>assistant<|end_header_id|>The output should be: {data_point['output']}"
    elif mode == 'test':
        text = f"{prefix_text}Here are the inputs: {data_point['input']} {instruction}\n<|start_header_id|>assistant<|end_header_id|>"
    
    return text

def load_train_data_for_llama3(TRAIN_SET,EOS_TOKEN,mode):
    raw_df = pd.read_csv(TRAIN_SET)
    df = raw_df[['PatientID','MRN','Narrative', 'jk_nec_features', 'jk_pneumatosis', 'jk_pvg','jk_freeair', 'notes']]
    df = pre_process_narrative(df)
    df['input'] =  df['Narrative'].apply(lambda x: generate_input_for_llama3(x))  
    df['output'] = df.apply(lambda row: generate_output(row['jk_nec_features'], row['jk_pneumatosis'], row['jk_pvg'], row['jk_freeair']), axis=1)
    results = df[['PatientID','MRN','input','output','jk_nec_features', 'jk_pneumatosis', 'jk_pvg','jk_freeair']]
    results['prompt'] = results.apply(lambda row: generate_prompt_for_llama3(row,mode), axis=1)
    results['prompt'] = results['prompt'].apply(lambda x: x + EOS_TOKEN)
    return results

def load_test_data_for_llama3(TEST_SET,EOS_TOKEN):
    raw_df = pd.read_csv(TEST_SET)
    df = raw_df[['PatientID','MRN','Narrative']]
    df = pre_process_narrative(df)
    df['input'] =  df['Narrative'].apply(lambda x: generate_input_for_llama3(x))  
    results = df[['PatientID','MRN','input']]
    results['prompt'] = results.apply(lambda row: generate_prompt_for_llama3(row,mode = 'test'), axis=1)
    results['prompt'] = results['prompt'].apply(lambda x: x + EOS_TOKEN)
    return results
