from transformers import RobertaTokenizer, RobertaModel
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import pickle
import argparse
import numpy as np
import os
from ast import literal_eval

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

argparser = argparse.ArgumentParser()
argparser.add_argument("dataset", choices=["published_writing", "twitter"])
argparser.add_argument("split", choices=["modern", "historical"])
argparser.add_argument("fname", type=str)

def get_roberta_embeddings(args):

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(device) # type: ignore

    if args.dataset == "twitter":
        dir = "data/tweet_roberta_tokenized"
    else:
        if args.split == "historical":
            dir = "data/coha_roberta_tokenized"
        else:
            dir = "data/coca_roberta_tokenized"

    with open(f'{dir}/{args.fname}.tsv') as f:
        rows = []
        for line in f.readlines():
            row = line.strip().split("\t")
            rows.append(row)
        df = pd.DataFrame(rows[1:], columns=rows[0])
    
    df = df.dropna().reset_index(drop=True)    

    embeddings = defaultdict(list)

    if args.dataset == "twitter":
        df_unique_texts = df.drop_duplicates(subset=["tweet_id", "sentence_id"]).reset_index(drop=True)
        print(f"Extracting RoBERTa embeddings for {len(df_unique_texts)} unique tweets...")
    else:
        df_unique_texts = df.drop_duplicates(subset=["file_id", "sentence_id"]).reset_index(drop=True)
        print(f"Extracting RoBERTa embeddings for {len(df_unique_texts)} unique sentences...")

    batch_size = 512
    n_batches = len(df_unique_texts) // batch_size
    for batch_idx in tqdm(range(n_batches + 1)):
        batch = df_unique_texts.iloc[batch_idx*batch_size : min((batch_idx+1)*batch_size, len(df_unique_texts))]
        with torch.no_grad():
            batch_texts = [" " + str(text) for text in batch["sentence"].tolist()]
            if args.dataset == "twitter":
                tweet_ids_and_sentence_ids = batch[["tweet_id", "sentence_id"]].values.tolist()
                positions = []
                target_words = []
                for tweet_id, sentence_id in tweet_ids_and_sentence_ids:
                    df_tweet = df[(df["tweet_id"] == tweet_id) & (df["sentence_id"] == sentence_id)]
                    positions.append([literal_eval(pos) for pos in df_tweet["position"].tolist()])
                    target_words.append(df_tweet["word"].tolist())
            else:
                file_ids_and_sentence_ids = batch[["file_id", "sentence_id"]].values.tolist()
                positions = []
                target_words = []
                for file_id, sentence_id in file_ids_and_sentence_ids:
                    df_sentence = df[(df["file_id"] == file_id) & (df["sentence_id"] == sentence_id)]
                    positions.append([literal_eval(pos) for pos in df_sentence["position"].tolist()])
                    target_words.append(df_sentence["word"].tolist())

            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            output = model(**encoded_input)
            hidden_states = output.last_hidden_state
            
            for text_idx in range(len(batch_texts)):
                positions_and_words = zip(positions[text_idx], target_words[text_idx])    
                encoded_text = encoded_input['input_ids'][text_idx]

                for target_position, target_word in positions_and_words:
                    if any([p >= 512 for p in target_position]):
                        continue

                    if not target_word == tokenizer.decode(encoded_text[target_position]).strip().lower():
                        print(f"Mismatch: {target_word} vs. {tokenizer.decode(encoded_text[target_position])}")
                        print(f"Skipping sentence: {batch_texts[text_idx]}")
                        continue

                    # Average the embeddings of the selected positions
                    if torch.cuda.is_available():
                        embedding = hidden_states[text_idx, target_position, :].mean(dim=0).detach().cpu().numpy()
                    else:
                        embedding = hidden_states[text_idx, target_position, :].mean(dim=0).detach().numpy()
                    embeddings[target_word].append(embedding)


    embedding_tensors = {}
    for word in embeddings:
        embedding_tensors[word] = np.vstack(embeddings[word])
    print(len(embeddings), "words with embeddings extracted.")
    os.makedirs(f'outputs/{args.dataset}/roberta_embeddings/', exist_ok=True)
    with open(f'outputs/{args.dataset}/roberta_embeddings/roberta_{args.split}_{args.fname}_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_tensors, f)

# ----------------------------------
if __name__ == "__main__":
    args = argparser.parse_args()
    get_roberta_embeddings(args)
