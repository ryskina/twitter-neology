import multiprocessing as mp
import time
import os

from nonword_filters import *
import json
from nltk.tokenize import TweetTokenizer, sent_tokenize

tokenizer = TweetTokenizer()

# Code for parallel processing by line from: 
# https://nurdabolatov.com/parallel-processing-large-file-in-python

def parallel_process(file_name):
    # Maximum number of processes we can run at a time
    cpu_count = mp.cpu_count() - 1

    file_size = os.path.getsize(file_name)
    chunk_size = file_size // cpu_count

    # Arguments for each chunk (eg. [('input.txt', 0, 32), ('input.txt', 32, 64)])
    chunk_args = []
    with open(file_name, 'r') as f:
        def is_start_of_line(position):
            if position == 0:
                return True
            # Check whether the previous character is EOL
            f.seek(position - 1)
            return f.read(1) == '\n'

        def get_next_line_position(position):
            # Read the current line till the end
            f.seek(position)
            f.readline()
            # Return a position after reading the line
            return f.tell()

        chunk_start = 0
        # Iterate over all chunks and construct arguments for `process_chunk`
        while chunk_start < file_size:
            chunk_end = min(file_size, chunk_start + chunk_size)

            # Make sure the chunk ends at the beginning of the next line
            while not is_start_of_line(chunk_end):
                chunk_end -= 1

            # Handle the case when a line is too long to fit the chunk size
            if chunk_start == chunk_end:
                chunk_end = get_next_line_position(chunk_end)

            # Save `process_chunk` arguments
            args = (file_name, chunk_start, chunk_end)
            chunk_args.append(args)

            # Move to the next chunk
            chunk_start = chunk_end

    with mp.Pool(cpu_count) as p:
        # Run chunks in parallel
        p.starmap(process_chunk, chunk_args)


def process_chunk(file_name, chunk_start, chunk_end):
    with open(file_name, 'r') as f:
        # Moving stream position to `chunk_start`
        f.seek(chunk_start)

        # Read and process lines until `chunk_end`
        for line in f:
            chunk_start += len(line)
            if chunk_start > chunk_end:
                break
            tokenize_line(line)


def measure(func, *args):
    time_start = time.time()
    result = func(*args)
    time_end = time.time()
    print(f'{func.__name__}: {time_end - time_start}')
    return result


def tokenize_line(line):
    tweet_json = json.loads(line)
    if "full_text" in tweet_json:
        text = tweet_json["full_text"]
    else:
        text = tweet_json["text"]
    for i, sentence in enumerate(sent_tokenize(text)):
        tokens = [token.lower() for token in tokenizer.tokenize(sentence) 
                if not is_hashtag(token)
                and not is_all_numbers_and_hyphens(token)
                and not is_all_emoji_punctuation_or_spaces_or_control(token)
                and not is_url(token)
                and is_all_alpha_num_hyphen_emoji(token)
                and len(token) <= 50
                and len(token) > 1
                ]
        if tokens:
            fout.write(f"{tweet_json['id']}\t{i+1}\t{' '.join(tokens)}\n")


#---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(f"data/tweet_tokenized/", exist_ok=True)
    for year in range(2007, 2022):
        print(f"Tokenizing tweets from year: {year}", flush=True)
        fout = open(f'data/tweet_tokenized/{year}.tsv', 'w+')
        measure(parallel_process, f'data/tweet_scraped_full/{year}_tweets.json')
