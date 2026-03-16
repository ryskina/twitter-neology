import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import os
from collections import defaultdict
import sys
from gensim.models import Word2Vec

def export_means(dataset, split):
    assert dataset in ["published_writing", "twitter"]
    assert split in ["historical", "modern"]

    running_counts = []
    running_means = []
    running_means_of_squares = []

    if dataset == "published_writing":
        if split == "historical":
            fnames = [f"contexts_part{part}" for part in range(1,11)]
            model_historical = Word2Vec.load(f"models/published_writing_word2vec/historical.w2v.bin")
            vocab = model_historical.wv.index_to_key
        else:
            fnames = ["contexts"]
            df_pairs = pd.read_csv(f"outputs/published_writing/all/pairs.csv")
            df_pairs_strict = pd.read_csv(f"outputs/published_writing/strict/pairs.csv")
            vocab = set(df_pairs["neologism"].tolist() + df_pairs_strict["neologism"].tolist())
    else:
        if split == "historical":
            fnames = [f"historical_contexts_part{part}" for part in range(1,101)]
            model_historical = Word2Vec.load(f"models/twitter_word2vec/historical.w2v.bin")
            vocab = model_historical.wv.index_to_key
        else:
            fnames = ["modern_contexts"]
            df_pairs = pd.read_csv(f"outputs/twitter/all/pairs.csv")
            df_pairs_strict = pd.read_csv(f"outputs/twitter/strict/pairs.csv")
            vocab = set(df_pairs["neologism"].tolist() + df_pairs_strict["neologism"].tolist())

    for fname in tqdm(fnames):
        all_embeddings = []
        with open(f"outputs/{dataset}/roberta_embeddings/roberta_{split}_{fname}_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            for w in data:
                if w in vocab:
                    all_embeddings.append(data[w])

        all_embeddings = np.concatenate(all_embeddings)
        mean_by_dim = np.mean(all_embeddings, axis=0)
        running_means.append(mean_by_dim)
        running_counts.append(len(all_embeddings))
        all_embeddings_squared = np.square(all_embeddings)
        running_means_of_squares.append(np.mean(all_embeddings_squared, axis=0))

    result = {
        "means": running_means,
        "means_of_squares": running_means_of_squares,
        "counts": running_counts
    }

    os.makedirs(f"models/{dataset}_roberta/", exist_ok=True)
    with open(f"models/{dataset}_roberta/{split}_stats.pkl", "wb") as f:
        pickle.dump(result, f)


def zscore_all_embeddings(dataset):
    assert dataset in ["published_writing", "twitter"]

    with open(f"models/{dataset}_roberta/historical_stats.pkl", "rb") as f:
        hist_stats = pickle.load(f)

    with open(f"models/{dataset}_roberta/modern_stats.pkl", "rb") as f:
        modern_stats = pickle.load(f)

    overall_mean = np.zeros(768)
    overall_mean_of_squares = np.zeros(768)
    total_count = np.sum(hist_stats["counts"] + modern_stats["counts"])

    for count, mean, mean_of_squares in zip(hist_stats["counts"], hist_stats["means"], hist_stats["means_of_squares"]):
        proportion = count / total_count
        overall_mean += mean * proportion
        overall_mean_of_squares += mean_of_squares * proportion
        
    for count, mean, mean_of_squares in zip(modern_stats["counts"], modern_stats["means"], modern_stats["means_of_squares"]):
        proportion = count / total_count
        overall_mean += mean * proportion
        overall_mean_of_squares += mean_of_squares * proportion

    overall_std = np.sqrt(overall_mean_of_squares - overall_mean**2)

    # Averaging historical embeddings

    counts_by_word = defaultdict(list)
    zscored_means_by_word = defaultdict(list)

    if dataset == "published_writing":
        fnames_historical = [f"contexts_part{part}" for part in range(1,11)]
        model_historical = Word2Vec.load(f"models/cocacoha_word2vec/historical.w2v.bin")
        historical_vocab = model_historical.wv.index_to_key
    else:
        fnames_historical = [f"historical_contexts_part{part}" for part in range(1,101)]
        model_historical = Word2Vec.load(f"models/twitter_word2vec/historical.w2v.bin")
        historical_vocab = model_historical.wv.index_to_key

    for fname in tqdm(fnames_historical):
        with open(f"outputs/{dataset}/roberta_embeddings/roberta_historical_{fname}_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            for w in data:
                if w in historical_vocab:
                    counts_by_word[w].append(len(data[w]))
                    zscored_mean = (np.mean(data[w], axis=0) - overall_mean) / overall_std
                    zscored_means_by_word[w].append(zscored_mean)

    print("Z-scoring historical embeddings...")
    final_zscored_embeddings = {}
    for word in tqdm(zscored_means_by_word):
        final_mean = np.zeros(768)
        counts = counts_by_word[word]
        zscored_means = zscored_means_by_word[word]
        for count, zscored_mean in zip(counts, zscored_means):
            proportion = count / sum(counts)
            final_mean += zscored_mean * proportion
        final_zscored_embeddings[word] = final_mean

    print(f"Historical embeddings computed for {len(final_zscored_embeddings)} words.")

    with open(f"models/{dataset}_roberta/historical_embeddings_zscored.pkl", "wb") as f:
        pickle.dump(final_zscored_embeddings, f)
    print("Historical embeddings saved.")

    # Averaging modern embeddings

    counts_by_word = defaultdict(list)
    zscored_means_by_word = defaultdict(list)

    if dataset == "published_writing":
        fnames_modern = ["contexts"]
        df_pairs = pd.read_csv(f"outputs/published_writing/all/pairs.csv")
        df_pairs_strict = pd.read_csv(f"outputs/published_writing/strict/pairs.csv")
        modern_vocab = set(df_pairs["neologism"].tolist() + df_pairs_strict["neologism"].tolist())
    else:
        fnames_modern = ["modern_contexts"]
        df_pairs = pd.read_csv(f"outputs/twitter/pairs.csv")
        df_pairs_strict = pd.read_csv(f"outputs/twitter_strict/pairs.csv")
        modern_vocab = set(df_pairs["neologism"].tolist() + df_pairs_strict["neologism"].tolist())

    print("Loading modern embeddings...")

    for fname in tqdm(fnames_modern):
        with open(f"files/{dataset}/roberta_embeddings/roberta_modern_{fname}_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            for w in tqdm(data):
                if w in modern_vocab:
                    counts_by_word[w].append(len(data[w]))
                    zscored_mean = (np.mean(data[w], axis=0) - overall_mean) / overall_std
                    zscored_means_by_word[w].append(zscored_mean)

    print("Z-scoring modern embeddings...")
    final_zscored_embeddings = {}
    for word in tqdm(zscored_means_by_word):
        final_mean = np.zeros(768)
        counts = counts_by_word[word]
        zscored_means = zscored_means_by_word[word]
        for count, zscored_mean in zip(counts, zscored_means):
            proportion = count / sum(counts)
            final_mean += zscored_mean * proportion
        final_zscored_embeddings[word] = final_mean

    print(f"Modern embeddings computed for {len(final_zscored_embeddings)} words.")

    with open(f"models/{dataset}_roberta/modern_embeddings_zscored.pkl", "wb") as f:
        pickle.dump(final_zscored_embeddings, f)
    print("Modern embeddings saved.")


# ----------------------------------
if __name__ == "__main__":
    dataset = sys.argv[1]
    assert dataset in ["published_writing", "twitter"]
    export_means(dataset, "historical")
    export_means(dataset, "modern")
    zscore_all_embeddings(dataset)