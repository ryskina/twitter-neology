import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import random
import os
import pickle
from utils import *
import json
from transformers import RobertaTokenizer
from nltk.tokenize import sent_tokenize, TweetTokenizer
import html
from gensim.models import Word2Vec


def get_all_ids_per_word(words, split):
    assert split in ["historical", "modern"]

    def process_lines(lines):
        ids_per_word = defaultdict(list)
        for line in lines:
            tweet_id, sentence_id, sentence = line.strip().split("\t")
            tokens = sentence.split()
            for word in set(tokens).intersection(words):
                ids_per_word[word].append((int(tweet_id), int(sentence_id)))
        return ids_per_word.items()

    if split == "historical":
        years = range(2007, 2011)
    else:
        years = range(2011, 2022)
    
    for year in years:
        with open(f"data/tweet_tokenized/{year}.tsv", "r") as fin:
            word_ids_pairs = parallelize_func(list(fin), process_lines, chunksz=100000, n_jobs=20)
                
        print("Combining results...")
        ids_per_word_year = defaultdict(list)
        for word, ids in tqdm(word_ids_pairs):
            ids_per_word_year[word].extend(ids)

        with open(f"outputs/twitter/ids_per_word_{year}.pkl", "wb") as fout:
            print(f"Saving tweet/sentence IDs per word for year {year}")
            pickle.dump(ids_per_word_year, fout)


# Sample a subset of tweet/sentence IDs for each word
def sample_sentences_per_word(words, split):
    assert split in ["historical", "modern"]
    lines_per_word = defaultdict(list)

    if split == "historical":
        years = range(2007, 2011)
        n = 250
        out_fname = "sample_ids_per_word"
    else:
        years = range(2011, 2022)
        n = 500
        out_fname = "sample_ids_per_neologism"

    for year in years:
        with open(f"outputs/twitter/ids_per_word_{year}.pkl", "rb") as fin:
            ids_per_word = pickle.load(fin)
            for word in words:
                lines_per_word[word].extend( (year, tweet_id, sentence_id) for tweet_id, sentence_id in ids_per_word[word] )  
            
        with open(f"outputs/twitter/{out_fname}.txt", "w+") as fout:
            for word in lines_per_word:
                random.seed(1234)
                if len(lines_per_word[word]) <= n:
                    sampled_lines = lines_per_word[word]
                else:
                    sampled_lines = random.sample(lines_per_word[word], n)

                for year, tweet_id, sentence_id in sampled_lines:
                    fout.write(f"{word}\t{year}\t{tweet_id}\t{sentence_id}\n")


def load_tweets_with_sampled_ids(split):
    assert split in ["historical", "modern"]
    if split == "historical":
        years = range(2007, 2011)
        in_fname = "sample_ids_per_word"
    else:
        years = range(2011, 2022)
        in_fname = "sample_ids_per_neologism"

    def check_ids_lines(lines):
        results = []
        for line in lines:
            tweet_json = json.loads(line)
            if int(tweet_json["id"]) in tweet_ids:
                results.append(line)
        return results

    fout = open(f"outputs/twitter/sample_{split}_tweets.jsonl", "w+")
    df = pd.read_csv(f"outputs/twitter/{in_fname}.txt", sep="\t", header=None, names=["word", "year", "tweet_id", "sentence_id"])

    for year in tqdm(years):
        print(f"Processing year {year}...")
        df_year = df[df["year"] == year]
        tweet_ids = df_year["tweet_id"].unique()
        with open(f"data/tweet_scraped_full/{year}_tweets.json", "r") as f:
            results = parallelize_func(list(f), check_ids_lines, chunksz=1000, n_jobs=100)
            for line in results:
                fout.write(line)

    fout.close()

    if split == "historical":
        # Splitting into parts for parallel processing in the next step
        n_parts = 100
        with open("outputs/twitter/sample_historical_tweets.jsonl", "r") as f:
            lines = f.readlines()
            part_size = len(lines) // n_parts
            for part in tqdm(range(1, n_parts+1)):
                if part < n_parts:
                    lines_part = lines[(part-1)*part_size:part*part_size]
                else:
                    lines_part = lines[(part-1)*part_size:]
                with open(f"outputs/twitter/sample_historical_tweets_by_part/sample_historical_tweets_part{part}.jsonl", "w+") as fout:
                    for line in lines_part:
                        fout.write(line)


def get_sentences_with_sampled_ids(split, part=None):
    os.makedirs(f"data/tweet_roberta_tokenized/", exist_ok=True)
    assert split in ["historical", "modern"]

    if split == "historical":
        ids_fname = "sample_ids_per_word"
        assert part is not None, "Please specify part number for historical split"
        in_fname = f"outputs/twitter/sample_historical_tweets_by_part/sample_historical_tweets_part{part}.jsonl"
        assert os.path.exists(in_fname), f"File {in_fname} does not exist"
    else:
        ids_fname = "sample_ids_per_neologism"
        in_fname = "outputs/twitter/sample_modern_tweets.jsonl"
        assert os.path.exists(in_fname), f"File {in_fname} does not exist"

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    df = pd.read_csv(f"outputs/twitter/{ids_fname}.txt", sep="\t", header=None, names=["word", "year", "tweet_id", "sentence_id"])

    fout = open(f"data/tweet_roberta_tokenized/{split}_contexts{'_part' + str(part) if part is not None else ''}.tsv", "w+")
    fout.write(f"word\tposition\ttweet_id\tsentence_id\tsentence\n")

    def extract_sentences_lines(lines):
        result = []
        for line in lines:
            tweet_json = json.loads(line)
            tweet_id = int(tweet_json["id"])
            df_subset = df[df["tweet_id"] == tweet_id]

            if df_subset.empty:
                print("Tweet id not found:", tweet_id)
                continue
            if "text" in tweet_json:
                text = tweet_json["text"]
            else:
                text = tweet_json["full_text"]

            sentences = sent_tokenize(text)
            for word, sentence_id in zip(df_subset["word"], df_subset["sentence_id"]):
                sentence = sentences[sentence_id - 1]
                sentence = html.unescape(sentence)
                sentence = sentence.replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace('&apos;', '').strip()
                tokenized_sentence = tokenizer.encode_plus(" " + sentence, add_special_tokens=True, truncation=True)['input_ids']

                words = TweetTokenizer().tokenize(sentence)
                word_occurrences = [occ for occ in words if occ.lower() == word]

                for occ in word_occurrences:
                    tokenized_occ = tokenizer.encode_plus(occ, add_special_tokens=False, truncation=True)['input_ids']
                    tokenized_occ_with_space = tokenizer.encode_plus(" " + occ, add_special_tokens=False, truncation=True)['input_ids']
                    # finding token positions
                    position = None
                    for i in range(len(tokenized_sentence) - min(len(tokenized_occ), len(tokenized_occ_with_space)) + 1):
                        if tokenized_sentence[i:i+len(tokenized_occ)] == tokenized_occ:
                            position = list(range(i, min(len(tokenized_sentence), i + len(tokenized_occ))))
                            break
                        elif tokenized_sentence[i:min(len(tokenized_sentence), i+len(tokenized_occ_with_space))] == tokenized_occ_with_space:
                            position = list(range(i, min(len(tokenized_sentence), i + len(tokenized_occ_with_space))))
                            break
                    if not position:
                        print(f"Could not find word '{word}' in sentence: {sentence}")
                        continue

                    if any([p >= 512 for p in position]):
                            print("Encountered position >= 512, skipping.")
                            print(word, sentence)
                            continue
                        
                    decoded = tokenizer.decode(tokenized_sentence[position[0]:position[-1]+1]).strip().lower()
                    assert decoded == word, f"Tokenization mismatch: {decoded} vs {word}"                    

                    result.append(f"{word}\t{position}\t{tweet_id}\t{sentence_id}\t{sentence}\n")
                    break
        return result

    with open(in_fname, "r") as f:
        results = parallelize_func(f.readlines(), extract_sentences_lines, chunksz=5000, n_jobs=20)
        fout.write("".join([line for res in results for line in res]))

    fout.close()


#---------------------------------------------------------------------------
if __name__ == '__main__':
    # Step 1: Get all tweet/sentence IDs for each word
    model_historical = Word2Vec.load(f"models/twitter_word2vec/historical.w2v.bin")
    historical_vocab = set(model_historical.wv.index_to_key)
    get_all_ids_per_word(historical_vocab, "historical")

    neologisms = set(pd.read_csv(f"outputs/published_writing/all/neologisms.txt", header=None)[0])
    get_all_ids_per_word(neologisms, "modern")

    # Step 2: Sample a subset of tweet/sentence IDs for each word
    sample_sentences_per_word(historical_vocab, "historical")
    sample_sentences_per_word(neologisms, "modern")

    # Step 3: Cache tweets with sampled IDs to avoid repeated file reading in the next step
    load_tweets_with_sampled_ids("historical")
    load_tweets_with_sampled_ids("modern")

    # Step 4: Extract sentences with sampled IDs and tokenize with RoBERTa tokenizer
    # For historical split, run separately for each part to parallelize
    for part in range(1, 101):
        get_sentences_with_sampled_ids("historical", part=part)
    
    get_sentences_with_sampled_ids("modern")