import pandas as pd
from tqdm import tqdm
import json
from scipy.stats import spearmanr
import numpy as np
from collections import Counter, defaultdict
from flair.nn import Classifier
from flair.data import Sentence


# Step 1: Filter words by minimum total count and first year of popular use
def neologism_candidates_by_popular_use():
    df = pd.read_csv("outputs/twitter/token_counts.csv", index_col="word")
    df["total"] = df.sum(axis=1)

    # Filtering all words by minimum total count
    min_count = 500
    df = df[df["total"] > min_count] 

    # Calculating the year of first popular use for each word
    alpha = 1/300
    df["first_popular_use"] = 2007
    df = df[df.index.notnull()]

    for word in tqdm(df.index):
        cumulative_usage = 0
        for year in range(2007, 2022):
            if pd.isna(df.at[word, str(year)]):
                continue
            cumulative_usage += df.at[word, str(year)]
            if cumulative_usage >= alpha * df.at[word, "total"]:
                df.at[word, "first_popular_use"] = year
                break

    # Selecting neologism candidates based on year of popular use
    df = df[df["first_popular_use"] >= 2011].sort_values("total", ascending=False)
    # df.to_csv("outputs/twitter/neologism_candidates_by_popular_use.csv")
    return df.reset_index()

# Step 1.5: extract all occurrences of candidate words in tweets, sample 100 occurrences per word, perform POS tagging on the samples
def pos_tag_candidates(df):
    # Finding all occurrences of each candidate word in tweets
    occurrences = []
    candidate_words = set(df["word"])
    tweet_ids_by_year = defaultdict(set)
    for year in tqdm(range(2007, 2022)):
        with open(f"data/tweet_tokenized/{year}.tsv") as fin:
            for line in fin:
                columns = line.strip().split('\t')
                assert len(columns) == 3
                tokens = columns[2].split()
                if candidate_words.intersection(tokens):
                    tweet_ids_by_year[year].add(columns[0])
                    for word in candidate_words.intersection(tokens):
                        occurrences.append({"year": year, "tweet_id": columns[0], "word": word})
        df_occurrences = pd.DataFrame(occurrences)

    df_occurrences.drop_duplicates(inplace=True)
    df_sample = pd.DataFrame()

    # Sampling 100 tweets with each candidate word
    for word in tqdm(candidate_words):
        df_word_filtered = df_occurrences[df_occurrences["word"] == word]
        if len(df_word_filtered) < 100:
            sample = df_word_filtered
            print("Less than 100 occurrences for", word)
        else:
            sample = df_occurrences[df_occurrences["word"] == word].sample(100)
        df_sample = pd.concat([df_sample, sample])

    # Performing POS tagging on the sampled tweets
    tagger = Classifier.load('pos')
    tag_counter_dict = defaultdict(Counter)
    for year in tqdm(range(2007, 2022)):
        tweet_ids = set(df_sample[df_sample["year"] == year]["tweet_id"])
        with open(f"data/tweet_scraped_full/{year}_tweets.json") as f:
            for line in f:
                item = json.loads(line)
                id = item['id']
                if isinstance(id, str): id = int(id)
                if id not in tweet_ids:
                    continue
                if "full_text" in item:
                    text = item['full_text']
                else:
                    text = item['text']
                try:
                    sentence = Sentence(text)
                    tagger.predict(sentence)
                    for token in sentence:
                        if token.text.lower() in candidate_words:
                            tag_counter_dict[token.text.lower()][token.tag] += 1
                except ValueError:
                    print(f"Error in {year}, {item}")

        with open("files/candidate-100-per-word-pos-count.jsonl", "w+") as f:
            for word in tag_counter_dict:
                f.write(json.dumps({'word': word, 'pos_counts_in_sample': dict(tag_counter_dict[word])}) + "\n")


# Step 2: Filter neologism candidates by POS tags
def filter_candidates_by_pos_tags(df):
    tags_to_remove = ["NNP", "CD", "FW", "NFP", "NNPS"]
    words_to_remove = []
    df.set_index("word", inplace=True)
    with open("outputs/twitter/candidate-100-per-word-pos-count.jsonl") as f:
        for line in f:
            item = json.loads(line)
            word = item["word"]
            if word not in df.index:
                continue
            pos_counter = Counter(item["pos_counts_in_sample"])
            top_count = pos_counter.most_common(1)[0][1]
            top_tag = pos_counter.most_common(1)[0][0]
            if top_tag in tags_to_remove:
                words_to_remove.append(word)
                continue
            # Discarding if count(NNP) == count(NN) or count(NNPS) == count(NNS)
            i = 1
            while pos_counter.most_common()[i-1][1] == top_count and i < len(pos_counter):
                tag = pos_counter.most_common()[i-1][0]
                if tag in tags_to_remove:
                    words_to_remove.append(word)
                    break
                i += 1

    df.drop(words_to_remove, inplace=True)
    # df.to_csv("outputs/twitter/neologism_candidates-nnp-fw-cd-nfp-removed.csv")
    return df.reset_index()

# Step 3: Filter out candidates with unusually peaked frequency distribution
def spearmanr_filter(df):
    df.set_index("word", inplace=True)
    unusual_distribution_words = []
    for word, row in df.iterrows():
        occurrences = [row[str(year)] for year in range(2007, 2022)]
        occurrences = np.asarray([int(c) if not pd.isna(c) else 0 for c in occurrences])
        if len(np.where(occurrences > 5)[0]) <= 3: # needs to occur enough times in at least 3 years
            if len(np.where(occurrences)) == 1 and occurrences[-1] > 0: # or just in the last year
                continue
            r = spearmanr(occurrences, range(15)).statistic # type: ignore
            if r < 0.5: # if distribution drops
                unusual_distribution_words.append(word)

    print(unusual_distribution_words)
    df.drop(unusual_distribution_words, inplace=True)
    # df.to_csv("outputs/twitter/neologism_candidates-spearmanr-nnp-fw-cd-nfp-removed.csv")
    return df.reset_index()

# ----------------------------------------------------------------
if __name__ == "__main__":
    df_candidates = neologism_candidates_by_popular_use()
    pos_tag_candidates(df_candidates)
    df_filtered_by_pos = filter_candidates_by_pos_tags(df_candidates)
    df_final = spearmanr_filter(df_filtered_by_pos)
    df_final["word"].to_csv("outputs/twitter/all/neologisms.txt", index=False, header=False)
