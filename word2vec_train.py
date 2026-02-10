from gensim.models import Word2Vec
import os
import argparse
import multiprocessing
import random

random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["twitter", "published_writing"],
                            help="Domain to train static embeddings on")
    parser.add_argument("data_split", choices=["historical", "modern"],
                            help="Historical or Modern data split")
    parser.add_argument("--embed_dim", type=int, default=300, 
                            help="Embedding dimensionality")
    parser.add_argument("--vocab_size", type=int, default=None, 
                            help="Maximum target vocabulary size (if not specified, min_count is set to 5 instead)")
    return parser.parse_args()

class SentenceIterator:
    def __init__(self, dataset, data_split):
        self.dataset = dataset
        assert data_split in ["modern", "historical"]
        self.data_split = data_split

    def __iter__(self):
        if self.dataset == "twitter":
            if self.data_split == 'historical':
                years = range(2007, 2011)
            else:
                years = range(2011, 2022)
            for year in years:
                print(f"Using tweets from year: {year}", flush=True)
                with open(f"data/tweet_tokenized/{year}.tsv") as fin:
                    for line in fin:
                        # not using the first column (tweet id) and the second (sentence number)
                        columns = line.strip().split('\t')
                        assert len(columns) == 3
                        tokens = columns[2].split()
                        yield tokens

        elif self.dataset == "published_writing":
            if self.data_split == 'historical':
                texts_path = "data/coha_tokenized"
            else:
                texts_path = "data/coca_tokenized"

            for fname in os.listdir(texts_path):
                if not fname.endswith(".tsv"):
                    continue
                print(f"Using texts from tokenized file: {fname}", flush=True)
                with open(f"{texts_path}/{fname}") as fin:
                    for line in fin:
                        columns = line.strip().split('\t')
                        assert len(columns) == 3
                        tokens = columns[2].split()
                        yield tokens

class EmbeddingTrainer:
    def __init__(self, params):
        self.dataset = params.dataset
        self.data_split = params.data_split
        self.embed_dim = params.embed_dim
        self.vocab_size = params.vocab_size

        os.makedirs(f"models/{self.dataset}_word2vec", exist_ok=True)
        self.model_file_path = f"models/{self.dataset}_word2vec/{self.data_split}.w2v.bin"
        self.sentences = []

    def train_w2v(self):
        """
        Learning Word2Vec embeddings from the provided data
        :return:
        """
        self.sentences = SentenceIterator(self.dataset, self.data_split)
        print(f"Building Word2Vec embeddings for {self.dataset.upper()} {self.data_split.upper()} data", flush=True)
        print(f"Model will be saved to file {self.model_file_path}", flush=True)

        if self.vocab_size:
            model = Word2Vec(self.sentences, vector_size=self.embed_dim, window=5, 
                max_final_vocab=self.vocab_size, workers=multiprocessing.cpu_count(), sg=1)
        else:
            model = Word2Vec(self.sentences, vector_size=self.embed_dim, window=5, 
                min_count=5, workers=multiprocessing.cpu_count(), sg=1)
        
        print(f"Finished training Word2Vec for {self.data_split.upper()} data", flush=True)
        model.save(self.model_file_path)
        print(f"Model saved to file", flush=True)

# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    trainer = EmbeddingTrainer(args)
    trainer.train_w2v()
