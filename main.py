import os
from extract_word_stats import NeologismControlMatcher
from extract_neighbourhood_stats import Word2VecNeighbourhoodStatsExtractor, RobertaNeighbourhoodStatsExtractor
import argparse
from gensim.models import Word2Vec
import pandas as pd

from utils import *
from word2vec_projection import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["published_writing", "twitter"])
    parser.add_argument("embedding_type", type=str, choices=["word2vec", "roberta"])
    parser.add_argument("--load_pairs", action='store_true', help="Load neologism-control pairs from file")
    parser.add_argument("--strict", action='store_true', help="Strictly filtered neologisms only")
    return parser.parse_args()


def main(params):
    dataset = params.dataset
    embedding_type = params.embedding_type

    model_path = f"models/{dataset}_{embedding_type}"
    if embedding_type == "word2vec":
        # Aligning embedding models
        model_historical = Word2Vec.load(f"{model_path}/historical.w2v.bin")
        model_modern = Word2Vec.load(f"{model_path}/modern.w2v.bin")
        model_modern_projected = smart_procrustes_align_gensim(model_historical, model_modern)
    else:
        embeddings_historical = pd.read_pickle(f"{model_path}/historical_embeddings_zscored.pkl")
        embeddings_modern = pd.read_pickle(f"{model_path}/modern_embeddings_zscored.pkl")
    
    # Collecting vocabulary
    historical_vocab = model_historical.wv.index_to_key if embedding_type == "word2vec" else list(embeddings_historical.keys())
    print(f"Loaded historical vocabulary with {len(historical_vocab)} words") # type: ignore
    modern_vocab = model_modern.wv.index_to_key if embedding_type == "word2vec" else list(embeddings_modern.keys())
    print(f"Loaded modern vocabulary with {len(modern_vocab)} words") # type: ignore

    # Loading neologisms
    print("Loading a list of neologisms...")
    neologism_file = f"outputs/{dataset}/{ 'strict' if params.strict else 'all' }/neologisms.txt"
    neologism_list = list(pd.read_csv(neologism_file, header=None)[0])    
    print("Done.")

    # Loading word frequencies
    print("Loading historical and modern word frequencies...")
    if dataset == "published_writing":
        historical_timesteps = ['1810s', '1820s', '1830s', '1840s', '1850s', '1860s', '1870s', '1880s', 
            '1890s', '1900s', '1910s', '1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s']
        modern_timesteps = ["modern"]
    else:
        historical_timesteps =  list(map(str, range(2007, 2011)))
        modern_timesteps =  list(map(str, range(2011, 2022)))
    overall_count_df = pd.read_csv(f'outputs/{dataset}/token_counts.csv', keep_default_na=False, na_values=[""]) 
    overall_count_df.set_index("word", inplace=True)    
    count_df = overall_count_df[historical_timesteps]
    overall_count_df["historical_total"] = overall_count_df[historical_timesteps].sum(axis=1)
    overall_count_df["modern_total"] = overall_count_df[modern_timesteps].sum(axis=1)
    overall_count_df["historical_total"] /= overall_count_df["historical_total"].sum()
    overall_count_df["modern_total"] /= overall_count_df["modern_total"].sum()
    count_df = count_df.fillna(0)
    timestep_totals = count_df.sum(axis=0)
    print("Done.")

    # Pairing neologisms with control words      
        
    pair_filename = f"outputs/{dataset}/{ 'strict' if params.strict else 'all'}/pairs.csv"

    if params.load_pairs or params.embedding_type == "roberta":
        print(f"Loading pairs of neologisms with relaxed control words...")
        if not params.load_pairs:
            print("(Re-matching neologisms to controls not available for RoBERTa embeddings.)")
        neologism_control_pairs = {}
        df_pairs = pd.read_csv(pair_filename)
        for _, row in df_pairs.iterrows():
            neologism_control_pairs[row["neologism"]] = row["control"]
    else:
        print(f"Pairing neologisms with relaxed control words...")
        ws = NeologismControlMatcher(model_historical, model_modern_projected, 
                                    count_df[count_df.index.isin(historical_vocab)], 
                                    overall_count_df[overall_count_df.index.isin(historical_vocab)]["historical_total"],
                                    overall_count_df[overall_count_df.index.isin(modern_vocab)]["modern_total"],
                                    timestep_totals)
        neologism_control_pairs = ws.pair_neologisms_with_controls(neologism_list, pair_filename)
    print("Done.")

    # Computing neighborhood density and frequency growth

    print("Estimating neighborhood density and frequency growth...")
    if embedding_type == "word2vec":
        outdir = f"outputs/{dataset}/{'strict' if params.strict else 'all'}/word2vec/"
        os.makedirs(outdir, exist_ok=True)
        ns = Word2VecNeighbourhoodStatsExtractor(model_historical, model_modern_projected, count_df[count_df.index.isin(historical_vocab)], timestep_totals)
    else:
        outdir = f"outputs/{dataset}/{'strict' if params.strict else 'all'}/roberta/"
        os.makedirs(outdir, exist_ok=True)
        ns = RobertaNeighbourhoodStatsExtractor(embeddings_historical, embeddings_modern, count_df[count_df.index.isin(historical_vocab)], timestep_totals)
    ns.compute_neighbourhood_stats_cosine(neologism_control_pairs, outdir)
    print("Done.")

# ----------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)
