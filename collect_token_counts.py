from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import os
import argparse
    
# Count the number of times each token appears in the dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["published_writing", "twitter"],
                            help="Domain to count tokens in")
    return parser.parse_args()

def get_counts_year_twitter(year):
    token_counts = Counter()
    with open(f"data/tweet_tokenized/{year}.tsv") as fin:
        for line in fin:
            columns = line.strip().split('\t')
            assert len(columns) == 3
            tokens = columns[2].split()
            token_counts.update(tokens)
    return token_counts

def get_counts_decade_coha(decade):
    token_counts = Counter()
    with open(f"data/coha_tokenized/{decade}.tsv") as fin:
        for line in fin:
            columns = line.strip().split('\t')
            assert len(columns) == 3
            tokens = columns[2].split()
            token_counts.update(tokens)
    return token_counts

def get_counts_genre_coca(genre):
    token_counts = Counter()
    with open(f"data/coca_tokenized/{genre}.tsv") as fin:
        for line in fin:
            columns = line.strip().split('\t')
            assert len(columns) == 3
            tokens = columns[2].split()
            token_counts.update(tokens)
    return token_counts

def main(params):
    os.makedirs(f"outputs/{params.dataset}", exist_ok=True)

    if params.dataset == "twitter":
        results = Parallel(n_jobs=15)(delayed(get_counts_year_twitter)(year) for year in range(2007, 2022))
        token_counts = dict(zip(range(2007, 2022), results))
    else:
        coha_decades = ['1810s', '1820s', '1830s', '1840s', '1850s', '1860s', '1870s', '1880s', 
                        '1890s', '1900s', '1910s', '1920s', '1930s', '1940s', '1950s', '1960s', 
                        '1970s', '1980s']
        print("Reading COHA files...")
        results_coha = Parallel(n_jobs=18)(delayed(get_counts_decade_coha)(decade) for decade in coha_decades)
        token_counts = dict(zip(coha_decades, results_coha))
        print("Reading COCA files...")
        results_coca = Parallel(n_jobs=5)(delayed(get_counts_genre_coca)(genre) for genre in 
                                           ['text_academic_rpe', 'text_fiction_awq', 
                                            'text_magazine_qch', 'text_newspaper_lsp'])
        token_counts["modern"] = sum(results_coca, Counter()) # type: ignore

    rows = []
    timestep_name = "year" if params.dataset == "twitter" else "decade"
    for timestep in tqdm(sorted(token_counts.keys())):
        for word in token_counts[timestep]: # type: ignore
            rows.append({timestep_name: timestep, "word": word, "count": token_counts[timestep][word]}) # type: ignore
    df = pd.DataFrame(rows)
    df = df.pivot(index="word", columns=timestep_name, values="count")
    print(df)
    df.to_csv(f"outputs/{params.dataset}/token_counts.csv")

# ----------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)