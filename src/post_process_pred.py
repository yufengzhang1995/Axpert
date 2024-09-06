import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('ignore')



def convert_ans2num(answer):
        mapping = {"Yes": 1, "No": 0, "Uncertain": 2, "None":0}
        return mapping.get(answer, -1) 

def process_answers(text):
    num_list = []
    # print(text)
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
    # print(num_list)
    return num_list


FILE_PATH = '/nfs/turbo/med-kayvan-lab/Projects/ARDS/Code/Yufeng/CXR/LLM/outputs/mistral/include_uncertain'
SAVE_PATH = '/nfs/turbo/med-kayvan-lab/Projects/ARDS/Code/Yufeng/CXR/LLM/pred_outputs/mistral/include_uncertain'
TEST = '/nfs/turbo/med-kayvan-lab/Projects/ARDS/Code/Yufeng/CXR/LLM/data/include_uncertain/df_test_imp.csv'
# TEST = '/nfs/turbo/med-kayvan-lab/Projects/ARDS/Code/Yufeng/CXR/LLM/LLM_infer_data/LLM_ready2review_df_0.6.csv'
test_df = pd.read_csv(TEST)
test_df = test_df[['PatientID','MRN','AccessionNumber','nec_features', 'pneumatosis', 'pvg','freeair']]
test_df = test_df.fillna(0)
test_pat_name = set(test_df.MRN)
print('The test file:',test_df.shape)

# test_size =  [0.4,0.5,0.6,0.7,0.8,0.9]
# test_size =  [0.2,0.4,0.6,0.8,1.0]
test_size =  [1.0]

for rs in [0,1,2]:
    for ts in test_size:
        file_path = os.path.join(FILE_PATH,f'ts_{ts}_rs_{rs}')
        file_names = os.listdir(file_path)
       
        save_path = os.path.join(SAVE_PATH,f'ts_{ts}_rs_{rs}')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for file in file_names:
            if ('csv' in file) and ('test_imp' in file) and ('epoch20' in file):#  and ('kw' in file)
                save_name = file.split('/')[-1].strip()
                print('save name:',save_name)
                FILE = os.path.join(file_path,file)
                df = pd.read_csv(FILE)
                otuput_pat_name = set(df.MRN)
                print('The output file from LLM is:', df.shape)
                df[['pred_nec', 'pred_pneumatosis', 'pred_pvg', 'pred_free_air']] = df['pred'].apply(lambda x: pd.Series(process_answers(x)))
                print('The output file from LLM is:', df.shape)
                df = pd.merge(df,test_df,on = ['AccessionNumber'])
                print('After merge:',df.shape)
                df.to_csv(os.path.join(save_path,f'pred_{save_name}'),index=False)
                print(f"*********Save to {os.path.join(save_path,f'pred_{save_name}')}*******")