import os
import re
from nltk import sent_tokenize
import random
from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizer
from gensim.models import Word2Vec
import pandas as pd

def sample_and_tokenize_sentences_roberta_coha(word_set):
    os.makedirs(f"data/coha_roberta_tokenized/", exist_ok=True)
    random.seed(1234)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    all_sentence_lines = []
    all_sentence_lines_per_word = defaultdict(list)
    all_found_words = set()
    for decade in tqdm(['1810s', '1820s', '1830s', '1840s', '1850s', '1860s', '1870s', '1880s', 
                   '1890s', '1900s', '1910s', '1920s', '1930s', '1940s', '1950s', '1960s', 
                   '1970s', '1980s']):
        for fname in os.listdir(f"data/COHA_text/{decade}"):
            if not fname.endswith(".txt"):
                continue
            with open(f"data/COHA_text/{decade}/{fname}") as fin:
                lines = fin.readlines()
                if len(lines) != 3:
                    print(f"File {decade}/{fname} contains {len(lines)} lines", flush=True)
                    continue
                texts = lines[2].split("\x00")
                i = 0
                for text in texts:
                    fragments = [frag.strip() for frag in text.split('@') if frag.strip()]
                    paragraphs = []
                    for fragment in fragments:
                        paragraphs += [par.strip() for par in fragment.split('<p>') if par.strip()]
                    for paragraph in paragraphs:
                        if not word_set.intersection(set(paragraph.lower().split())):
                            continue
                        sents = sent_tokenize(paragraph)
                        for sent in sents:
                            sent = " " + sent + " "
                            sent = sent.replace(" n't", "n't").replace(" 's", "'s").replace(" ’s", "’s")
                            sent = sent.replace(" 're", "'re").replace(" 've", "'ve").replace(" 'd", "'d")
                            sent = sent.replace(" 'll", "'ll").replace(" 'm", "'m")
                            found_words = word_set.intersection(set(sent.lower().split()))
                            all_found_words.update(found_words)
                            if not found_words:
                                continue
                            all_sentence_lines.append(f"{fname}\t{i+1}\t{sent.strip()}\n")
                            for word in found_words:
                                all_sentence_lines_per_word[word].append(len(all_sentence_lines)-1)
                            i += 1

    all_found_words = list(all_found_words)
    n_words = len(all_found_words)
    part_size = n_words // 10
    for part in range(10):
        print(f"Sampling part {part+1}/10")
        with open(f"data/coha_roberta_tokenized/contexts_part{part+1}.tsv", "w+") as fout:
            fout.write("word\tposition\tfile_id\tsentence_id\tsentence\n")
            if part == 9:
                end_idx = n_words
            else:
                end_idx = (part+1)*part_size
            for word in all_found_words[part*part_size: end_idx]:
                sentence_line_ids = all_sentence_lines_per_word[word]
                if len(sentence_line_ids) <= 250:
                    sampled_lines = [all_sentence_lines[i] for i in sentence_line_ids]
                else:
                    sampled_lines = [all_sentence_lines[i] for i in random.sample(sentence_line_ids, 250)]
                for line in sampled_lines:
                    sentence = line.strip().split('\t')[-1]
                    tokenized_sentence = tokenizer.encode_plus(" " + sentence, add_special_tokens=True)['input_ids']

                    word_occurrences = [occ for occ in sentence.strip().split() if occ.lower() == word]

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

                        fout.write(f"{word}\t{position}\t{line}")
                        break


def sample_and_tokenize_sentences_roberta_coca(neologism_set):
    os.makedirs(f"data/coca_roberta_tokenized/", exist_ok=True)
    random.seed(1234)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    all_sentence_lines = []
    all_sentence_lines_per_word = defaultdict(list)
    all_found_words = set()
    for subdir in tqdm(['text_academic_rpe', 'text_fiction_awq', 'text_magazine_qch', 'text_newspaper_lsp']):
        for fname in os.listdir(f"data/COCA_text/{subdir}"):
            if not fname.endswith(".txt"):
                continue
            with open(f"data/COCA_text/{subdir}/{fname}") as fin:            
                texts = []
                for line in fin.readlines()[1:]:
                    line = line[line.find(' '):] # remove the first ID token
                    for text in line.split("\x00"):
                        # skipping texts that do not contain any neologisms
                        if len(neologism_set.intersection(set(text.lower().split()))) > 0:
                            texts.append(text)

                i = 0
                for text in texts:
                    fragments = [frag.strip() for frag in text.split('@') if frag.strip()]
                    paragraphs = []
                    for fragment in fragments:
                        paragraphs += [par.strip() for par in fragment.split('<p>') if par.strip()]
                    for paragraph in paragraphs:    
                        if not neologism_set.intersection(set(paragraph.lower().split())):
                            continue
                        sents = sent_tokenize(paragraph)
                        for sent in sents:
                            found_words = neologism_set.intersection(set(sent.lower().split()))
                            all_found_words.update(found_words)
                            if not found_words:
                                continue
                            all_sentence_lines.append(f"{fname}\t{i+1}\t{sent.strip()}\n")
                            for word in found_words:
                                all_sentence_lines_per_word[word].append(len(all_sentence_lines)-1)
                            i += 1

    print(len(all_found_words), "neologisms found in COCA texts.")
    print(f"Sampling...")
    with open(f"data/coca_roberta_tokenized/contexts.tsv", "w+") as fout:
        fout.write("word\tposition\tfile_id\tsentence_id\tsentence\n")
        for word in all_found_words:
            sentence_line_ids = all_sentence_lines_per_word[word]
            if len(sentence_line_ids) <= 500:
                sampled_lines = [all_sentence_lines[i] for i in sentence_line_ids]
            else:
                sampled_lines = [all_sentence_lines[i] for i in random.sample(sentence_line_ids, 500)]
            for line in sampled_lines:
                sentence = line.strip().split('\t')[-1]
                tokenized_sentence = tokenizer.encode_plus(" " + sentence, add_special_tokens=True, truncation=True)['input_ids']

                word_occurrences = [occ for occ in sentence.strip().split() if occ.lower() == word]

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

                    fout.write(f"{word}\t{position}\t{line}")
                    break


# ---------------------------------------------------------------------------
if __name__ == '__main__':

    model_historical = Word2Vec.load(f"models/published_writing_word2vec/historical.w2v.bin")
    historical_vocab = model_historical.wv.index_to_key
    sample_and_tokenize_sentences_roberta_coha(set(historical_vocab))

    neologism_set = set(pd.read_csv(f"outputs/published_writing/all/neologisms.txt", header=None)[0])
    sample_and_tokenize_sentences_roberta_coca(neologism_set)
