This repository contains the code and word lists for the [paper](https://arxiv.org/abs/2602.13123):
```
From sunblock to softblock: Analyzing the correlates of neology in published writing and on social media
Maria Ryskina, Matthew R. Gormley, Kyle Mahowald, David R. Mortensen, Taylor Berg-Kirkpatrick, Vivek Kulkarni
LChange 2026
```

Please contact mryskina@alumni.cmu.edu for any questions.

## Data

### Source datasets

* **Published writing**: This code uses the [COHA](https://www.english-corpora.org/coha/) and [COCA](https://www.english-corpora.org/coca/) corpora in pre-tokenized plain text format. The corpora need to be obtained from https://www.corpusdata.org/. The code in `pub_tokenize.py` and `pub_roberta_sample_and_tokenize.py` assumes that the data is stored in `data/COHA_text/` and `data/COCA_text/` respectively.
* **Twitter**: The tweet IDs for each year provided in `data/tweet_ids.txt`. To access the full tweet text, the tweets need to be rehydrated via the Twitter/X API. The code in `twt_tokenize.py` and `twt_roberta_sample_and_tokenize.py` assumes that the tweets for a given year are stored in `data/tweet_scraped_full/{year}_tweets.jsonl`, with each line in the following format:
```
{"id": XXXXXX, "full_text": "Full text of tweet #XXXXXX"}
```
### Neologism lists 

* For published writing, the full list of neologism candidates (`outputs/published_writing/all/neologisms.txt`)  is taken from [our previous work](https://github.com/ryskina/neology/blob/master/files/neologisms.txt) [(Ryskina et al., 2020)](https://aclanthology.org/2020.scil-1.43.pdf). 
* For Twitter, neologism candidates (`outputs/twitter/all/neologisms.txt`) are extracted using the code in `twt_identify_neologisms.py`.
* The neologism candidates are then manually filtered (Appendix B.3). The resulting final neologism lists can be found at (`outputs/{published_writing|twitter}/strict/neologisms.txt`). All results reported in the main body of the paper are for the manually filtered lists of neologisms.  

## Code

### Preprocessing/extracting embeddings
* `{pub|twt}_tokenize.py`: tokenizes the corresponding dataset (published writing or tweets) into sentences and tokens. The resulting data is used to estimate token counts, identify neologisms (for the Twitter domain), and train Word2Vec models. For tweets, additional token filtering is applied (see Appendix B.1). The resulting files are stored in `data/coha_tokenized/` (published writing, historical split), `data/coca_tokenized/` (published writing, modern split), and `data/tweet_tokenized/` (tweets, both splits).
* `collect_token_counts.py`: uses the tokenized data to estimate token counts at each time step (decade for published writing, year for Twitter).
* `twt_identify_neologisms.py`: using the obtained token counts, estimates the first year of popular use for each word to identify a set of candidate neologisms. Additional filtering by POS tag and distribution shape is applied (see Appendix B.2). The resulting word list can be found at `outputs/twitter/all/neologisms.txt`.
* `word2vec_train.py`: learns Word2Vec embeddings for the chosen dataset and split using [Gensim](https://radimrehurek.com/gensim/). The trained models are saved to `models/{published_writing|twitter}_word2vec/{historical|modern}.w2v.bin`.
* `word2vec_projection.py`: projects the modern Word2Vec embeddings into the historical space. It is based heavily on [Ryan Heuser's Gensim port](https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf) of William Hamilton's alignment code in [HistWords](https://github.com/williamleif/histwords). 
* `{pub|twt}_roberta_sample_and_tokenize.py`: given a set of words (full vocabulary for the historical split, list of neologisms for the modern split), samples a subset of sentences with each given word. The sentences will be used for estimating the average [RoBERTa](https://huggingface.co/FacebookAI/roberta-base) embedding of each word. The resulting files are stored in `data/coha_roberta_tokenized/` (published writing, historical split), `data/coca_roberta_tokenized/` (published writing, modern split), and `data/tweet_roberta_tokenized/` (tweets, both splits).
* `collect_roberta_embeddings.py`: for each sampled sentence, extract the mean-pooled hidden state corresponding to the target word from the last layer of RoBERTa. The resulting `.pkl` files are stored in `outputs/{published_writing|twitter}/roberta_embeddings/`.
* `average_roberta_embeddings.py`: computes the average z-scored RoBERTa embedding for each word. The resulting embeddings are stored at `models/{published_writing|twitter}_roberta/{historical|modern}_embeddings_zscored.pkl`.

Embeddings and intermediate output files can be shared upon request.

### Main analysis script
```
python main.py dataset embedding_type [--strict] [--load_pairs]
```
where:
* `dataset` specifies the domain (`published_writing` or `twitter`).
* `embedding_type` specifies whether to use static (`word2vec`) or contextual (`roberta`) embeddings.
* `--strict`: if used, only the manually filtered lust of neologisms will be used in the analysis (`outputs/{published_writing|twitter}/strict/neologisms.txt`). Otherwise, the full, automatically extracted list of candidates will be used (`outputs/{published_writing|twitter}/all/neologisms.txt`).
* `--load_pairs`: if used, neologism--control pairs will be loaded from file (`outputs/{published_writing|twitter}/{all|strict}/pairs.csv`). Otherwise, the pairs are recomputed from scratch (available only for Word2Vec embeddings).

The results are saved to `outputs/{published_writing|twitter}/{all|strict}/{roberta|word2vec}_results.csv`.