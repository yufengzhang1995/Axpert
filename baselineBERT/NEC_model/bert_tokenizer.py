import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
import json
from tqdm import tqdm
import argparse
import re 

def extract_impression_or_findings(text):
        pattern = re.compile(r'(impression|finding|findings|impressions)(.*)', re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
                section = match.group(2).strip()
                return section
        else:   
                print("No impression or findings, return full text")
                return text
def get_impressions_from_csv(path):
        df = pd.read_csv(path)
        imp = df['Narrative'].apply(lambda x: extract_impression_or_findings(x))
        # imp.to_csv('imp.csv')
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()
        df['impression'] = imp
        return imp,df

def tokenize(impressions, tokenizer):
        new_impressions = []
        print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
        for i in tqdm(range(impressions.shape[0])):
                tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
                if tokenized_imp: #not an empty report
                        res = tokenizer.encode_plus(tokenized_imp)['input_ids']
                        if len(res) > 512: #length exceeds maximum size
                                #print("report length bigger than 512")
                                res = res[:511] + [tokenizer.sep_token_id]
                        new_impressions.append(res)
                else: #an empty report
                        new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
        return new_impressions

def load_list(path):
        with open(path, 'r') as filehandle:
                impressions = json.load(filehandle)
                return impressions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize radiology report impressions and save as a list.')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                        under the \"Report Impression\" column')
    parser.add_argument('-tk', '--tk', type=str, nargs='?', default = 'bert-base-uncased',
                        help='path to intended output file')
    parser.add_argument('-odf', '--odf', type=str, nargs='?', required=True,
                        help='path to intended output file')
    parser.add_argument('-ojson', '--ojson', type=str, nargs='?', required=True,
                        help='path to intended output file')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.ojson
    df_output_path = args.odf
    tk = args.tk
    print(tk)
    
    tokenizer = BertTokenizer.from_pretrained(tk)

    impressions,df = get_impressions_from_csv(csv_path)
    
    df.to_csv(df_output_path)
    new_impressions = tokenize(impressions, tokenizer)
    with open(out_path, 'w') as filehandle:
            json.dump(new_impressions, filehandle)
            
