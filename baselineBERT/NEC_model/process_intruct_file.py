
import numpy as np
import pandas as pd
import os,sys
import re

def extract_input(prompt):
    start_text = "Q: This is a pediatric radiology report: "
    end_text = "\n(1) Does the child have necrotizing enterocolitis?\n"
    start_pos = prompt.find(start_text) + len(start_text)
    end_pos = prompt.find(end_text)
    

    input_text = prompt[start_pos:end_pos]
    return input_text


def convert_ans2num(answer):
        mapping = {"Yes": 1, "No": 0, "Uncertain": 2, "None":0}
        return mapping.get(answer, 0) 

def process_answers(text):
    num_list = []
    answer_book = re.findall(r"model <answer>(.*?)</answer>", text, re.DOTALL) # for 0.6
    if answer_book:
        answer_list = answer_book[0].strip().split('\n')
        for answer in answer_list:
            answer = answer.split('.')[-1].strip()
            # print(answer)
            num = convert_ans2num(answer)
            num_list.append(num)
    else:
        num_list = [0,0,0,0]
    
    if len(num_list) > 4:
        num_list = num_list[0:4]
    
    return num_list

def find_file_recursively(root_dir):
    path_ls = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'csv' in filename:
                path_ls.append(os.path.join(dirpath, filename))
    return path_ls

# Root directory and target file name
root_directory = 
file_paths = find_file_recursively(root_directory)
print(file_paths)


for path in file_paths:
    df = pd.read_csv(path,index_col = 0)
    print(f'Read from : {path} ************')
    df['Narrative'] = df['prompt'].apply(lambda x: extract_input(x))
    try:
        df[['nec_features', 'pneumatosis', 'pvg', 'freeair']] = df['pred'].apply(lambda x: pd.Series(process_answers(x)))
        print(df.shape)
        print(df.columns)
        df.to_csv(path)
        print(f'Save to : {path} ************')
    except ValueError as e:
        print(f"Error reading {path}: {e}")
    

