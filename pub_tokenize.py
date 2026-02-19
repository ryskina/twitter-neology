import os
from nltk import sent_tokenize
from joblib import Parallel, delayed


def tokenize_decade_coha(decade):
    print("Tokenizing COHA texts from decade:", decade, flush=True)
    os.makedirs(f"data/coha_tokenized/", exist_ok=True)
    with open(f"data/coha_tokenized/{decade}.tsv", "w+") as fout:
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
                        sents = sent_tokenize(paragraph)
                        for sent in sents:
                            tokens = [token.lower() for token in sent.strip().split() if token]
                            fout.write(f"{fname}\t{i+1}\t{' '.join(tokens)}\n")
                            i += 1


def tokenize_genre_coca(genre):
    print("Tokenizing COCA texts from genre:", genre, flush=True)
    os.makedirs(f"data/coca_tokenized/", exist_ok=True)
    with open(f"data/coca_tokenized/{genre}.tsv", "w+") as fout:      
        for fname in os.listdir(f"data/COCA_text/{genre}"):
            if not fname.endswith(".txt"):
                continue
            with open(f"data/COCA_text/{genre}/{fname}") as fin:
                lines = fin.readlines()[1:]
                texts = []
                for line in lines:
                    texts += line.split("\x00")
                i = 0
                for text in texts:
                    fragments = [frag.strip() for frag in text.split('@') if frag.strip()]
                    paragraphs = []
                    for fragment in fragments:
                        paragraphs += [par.strip() for par in fragment.split('<p>') if par.strip()]
                    for paragraph in paragraphs:    
                        sents = sent_tokenize(paragraph)
                        for sent in sents:
                            tokens = [token.lower() for token in sent.strip().split() if token]
                            fout.write(f"{fname}\t{i+1}\t{' '.join(tokens)}\n")
                            i += 1


# ---------------------------------------------------------------------------
if __name__ == '__main__':

    coha_decades = ['1810s', '1820s', '1830s', '1840s', '1850s', '1860s', '1870s', '1880s', 
                   '1890s', '1900s', '1910s', '1920s', '1930s', '1940s', '1950s', '1960s', 
                   '1970s', '1980s']
    Parallel(n_jobs=len(coha_decades))(delayed(tokenize_decade_coha)(decade) for decade in coha_decades)
    
    coca_genres = ['text_academic_rpe', 'text_fiction_awq', 'text_magazine_qch', 
                    'text_newspaper_lsp']
    Parallel(n_jobs=len(coca_genres))(delayed(tokenize_genre_coca)(genre) for genre in coca_genres)

