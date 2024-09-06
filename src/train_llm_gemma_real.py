from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextStreamer
import re
import os
import random
from AXRdataset import *
import argparse
from datasets import Dataset
from sklearn.model_selection import train_test_split
import time

import warnings
warnings.filterwarnings("ignore")

def pre_process_narrative(df):
    imp = df['Narrative']
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True).replace('\s+', ' ', regex=True).str.strip()
    df['Narrative'] = imp
    return df

def generate_input(input):
    prompt = f"""Q: This is a pediatric radiology report: {input}\n"""
    prompt += "(1) Does the child have necrotizing enterocolitis?\n   A. Yes\n   B. No\n   C. Uncertain\n"
    prompt += "(2) Does the child have pneumatosis?\n   A. Yes\n   B. No\n   C. Uncertain\n"
    prompt += "(3) Does the child have portal venous gas?\n   A. Yes\n   B. No\n   C. Uncertain\n"
    prompt += "(4) Does the child have free air?\n   A. Yes\n   B. No\n   C. Uncertain\n"
    return prompt


def generate_instruction():
    return 'Please answer the four numbered questions. Write the answers using A, B, or C in between <answer> tags.'
    
def get_ans(x):
    if x == 1:
        return 'A. Yes'
    elif x == 2:
        return 'C. Uncertain'
    elif x == 0:
        return 'B. No'

def generate_output(jk_nec_features,jk_pneumatosis,jk_pvg,jk_freeair):
    temp = np.array([jk_nec_features,jk_pneumatosis,jk_pvg,jk_freeair])
    results = [get_ans(i) for i in temp]     
    answer = '<answer>\n'
    for result in results:
        answer += f' {result}\n'  
    answer += '</answer>'
    return answer

def generate_prompt_for_gemma(data_point, mode):
    # Generate prompt
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                  'appropriately completes the request.\n\n'
    
    if mode in ['train', 'eval']:
        text = (f"<start_of_turn>user {prefix_text} {data_point['instruction']} here are the inputs {data_point['input']} <end_of_turn>\n"
                f"<start_of_turn>model {data_point['output']} <end_of_turn>")
    elif mode == 'test':
        text = (f"<start_of_turn>user {prefix_text} {data_point['instruction']} here are the inputs {data_point['input']} <end_of_turn>\n")
    
    return text


def load_train_data_for_gemma(TRAIN_SET,EOS_TOKEN,mode,test_size = 1,random_state = 42):
    raw_df = pd.read_csv(TRAIN_SET)
    df = raw_df[['PatientID','MRN','AccessionNumber','Narrative', 'nec_features', 'pneumatosis', 'pvg','freeair']]
    df.fillna(0, inplace=True)
    df = pre_process_narrative(df)
    df['input'] =  df['Narrative'].apply(lambda x: generate_input(x))  
    df['instruction'] =  generate_instruction()
    df['output'] = df.apply(lambda row: generate_output(row['nec_features'], row['pneumatosis'], row['pvg'], row['freeair']), axis=1)
    results = df[['PatientID','MRN','AccessionNumber','input','instruction','output','nec_features', 'pneumatosis', 'pvg','freeair']]
    results['prompt'] = results.apply(lambda row: generate_prompt_for_gemma(row,mode), axis=1)
    # results['prompt'] = results['prompt'].apply(lambda x: x + EOS_TOKEN)
    
    if test_size != 1:
        # results['stratify_col'] = results.apply(lambda row: f"{row['nec_features']}_{row['pneumatosis']}_{row['pvg']}_{row['freeair']}", axis=1)
        _, test_df = train_test_split(results, test_size=test_size, stratify=results['nec_features'], random_state=random_state)
        # train_df = train_df.drop(columns=['stratify_col'])
        return test_df
    else:
        return results

def load_test_data_for_gemma(TEST_SET,EOS_TOKEN,add_gt = False):
    raw_df = pd.read_csv(TEST_SET)
    if add_gt:
        df = raw_df[['PatientID','MRN','AccessionNumber','Narrative', 'nec_features', 'pneumatosis', 'pvg','freeair']]
        df.fillna(0, inplace=True)
        df['gt'] = df.apply(lambda row: generate_output(row['nec_features'], row['pneumatosis'], row['pvg'], row['freeair']), axis=1)
    else:
        df = raw_df[['PatientID','MRN','AccessionNumber','Narrative']]
    df = pre_process_narrative(df)
    df['input'] =  df['Narrative'].apply(lambda x: generate_input(x))  
    df['instruction'] =  generate_instruction()
    if add_gt:
        results = df[['PatientID','MRN','AccessionNumber','input','instruction','gt']]
    else:
        results = df[['PatientID','MRN','AccessionNumber','input','instruction']]
    results['prompt'] = results.apply(lambda row: generate_prompt_for_gemma(row,mode = 'test'), axis=1)
    # results['prompt'] = results['prompt'].apply(lambda x: x + EOS_TOKEN)
    return results



def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description = 'LLM model training')
    parser.add_argument('--model_name', type = str, default = "unsloth/gemma-7b-it-bnb-4bit", help = 'model_name')
    parser.add_argument('--epoch', type = int, default = 2, help = 'epoch')
    parser.add_argument('--lr',type = float, default = 2e-4, help = 'learning rate')
    parser.add_argument('--save_dir',  type = str)
    parser.add_argument('--train_set',  type = str)
    parser.add_argument('--test_set',  type = str)
    parser.add_argument('--checkpoint',  type = str)
    # --inference to default to False and only be True when explicitly specified:
    parser.add_argument('--inference', action='store_true', help='Toggle inference mode on')
    parser.add_argument('--test_size',  type = float, default = 1.0)
    parser.add_argument('--seed',  type = int, default = 42)
    
    args = parser.parse_args()
    
    LR = args.lr
    max_seq_length = 2048 
    dtype = None 
    load_in_4bit = True 
    N_EPOCH = args.epoch
    LR = 2e-4
    model_name = args.model_name
    test_size = args.test_size
    random_state = args.seed
    SAVE_DIR = os.path.join(args.save_dir,f'ts_{test_size}_rs_{random_state}')
    TRAIN_SET = args.train_set
    TEST_SET = args.test_set
    CHECKPOINT_DIR = os.path.join(args.checkpoint,f'ts_{test_size}_rs_{random_state}_epoch_{N_EPOCH}')
    infer = args.inference
    print('infer?',infer)
    
    
    fourbit_models = [
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "unsloth/llama-2-7b-bnb-4bit",
            "unsloth/gemma-7b-bnb-4bit",
            "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
            "unsloth/gemma-2b-bnb-4bit",
            "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
] 
    
    # load pre-trained model
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit)
    if not infer:
        # get peft model
        print('loading peft model.....')
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, 
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, 
            bias = "none",   
            use_gradient_checkpointing = True,
            random_state = 3407,
            use_rslora = False,  
            loftq_config = None)
    
    EOS_TOKEN = tokenizer.eos_token
    
    if not infer:
        # load training dataset
        print('loading training data.....') 
        train_df = load_train_data_for_gemma(TRAIN_SET,EOS_TOKEN,mode = 'train',test_size = test_size,random_state = random_state)
        train_df.to_csv('train_ss.csv')
        print('The training data size is:',train_df.shape[0])
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = train_dataset.shuffle(seed=1234)  # Shuffle dataset here
        train_dataset = train_dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
        
        
        # set trainer
        trainer = SFTTrainer(
                    model = model,
                    tokenizer = tokenizer,
                    train_dataset = train_dataset,
                    dataset_text_field = "prompt",
                    max_seq_length = max_seq_length,
                    dataset_num_proc = 2,
                    packing = False, 
                    args = TrainingArguments(
                            per_device_train_batch_size = 2,
                            gradient_accumulation_steps = 4,
                            warmup_steps = 5,
                            num_train_epochs=N_EPOCH,
                            learning_rate = LR,
                            fp16 = not torch.cuda.is_bf16_supported(),
                            bf16 = torch.cuda.is_bf16_supported(),
                            logging_steps = 1,
                            optim = "adamw_8bit",
                            weight_decay = 0.01,
                            lr_scheduler_type = "linear",
                            seed = 3407,
                            output_dir = SAVE_DIR,
            ),
        )

        trainer_stats = trainer.train()
    
    # load test dataset
    print('loading test data.....')
    EOS_TOKEN = tokenizer.eos_token 

    if not infer:
        test_df = load_test_data_for_gemma(TEST_SET ,EOS_TOKEN,add_gt = True)
    else:
        test_df = load_test_data_for_gemma(TEST_SET ,EOS_TOKEN,add_gt = False)
    
    # inference mode
    FastLanguageModel.for_inference(model) 
    preds = []
    time_records = []
    for prompt in test_df['prompt']:
        inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")
        text_streamer = TextStreamer(tokenizer)
        start_time = time.time()
        outputs = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)
        pred = tokenizer.batch_decode(outputs)[0]
        end_time = time.time()
        execution_time = end_time - start_time
        time_records.append(execution_time)
        preds.append(pred)
    time_records = np.array(time_records)
    print('the test datasize is: ',test_df.shape)
    print(f'Inference time: mean and std are {np.mean(time_records)} and {np.std(time_records)}.')
    
    
    test_df['pred'] = preds

    save_model_name = model_name.split('/')[-1].strip()
    save_set_name = TEST_SET.split('/')[-1].split('.')[0].strip()
    print(save_model_name)
    print(save_set_name)
    save_name = '_'.join([save_set_name, save_model_name])
    if infer:
        test_df.to_csv(os.path.join(SAVE_DIR,f'{save_name}.csv'), index=False)
    
    if not infer:
        test_df.to_csv(os.path.join(SAVE_DIR,f'{save_name}_epoch{N_EPOCH}_lr{LR}.csv'), index=False)
        model.save_pretrained(os.path.join(CHECKPOINT_DIR,f"{save_model_name}_epoch{N_EPOCH}_lr{LR}")) 
    
if __name__ == '__main__':
    main()
    
