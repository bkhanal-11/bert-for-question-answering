from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from pathlib import Path
import os
import wget
import zipfile

MAX_LEN = 64

def download_and_extract_data():
    # Download the data from the URL
    url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    wget.download(url, './')

    # Extract the data from the zip file
    with zipfile.ZipFile('cornell_movie_dialogs_corpus.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
    
    # Delete the zip file
    os.remove('cornell_movie_dialogs_corpus.zip')
    
    # Create the datasets directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # Move the necessary files to the datasets directory
    conv_path_name = 'datasets/movie_conversations.txt'
    line_path_name = 'datasets/movie_lines.txt'

    os.rename('cornell movie-dialogs corpus/movie_conversations.txt', conv_path_name)
    os.rename('cornell movie-dialogs corpus/movie_lines.txt', line_path_name)

    return conv_path_name, line_path_name


def get_dialogue_pairs(movie_conv, movie_lines):
    # loading all data into memory
    with open(movie_conv, 'r', encoding='iso-8859-1') as c:
        conv = c.readlines()
    with open(movie_lines, 'r', encoding='iso-8859-1') as l:
        lines = l.readlines()

    # splitting text using special lines
    lines_dic = {}
    for line in lines:
        objects = line.split(" +++$+++ ")
        lines_dic[objects[0]] = objects[-1]

    # generate question answer pairs
    pairs = []
    for con in conv:
        ids = eval(con.split(" +++$+++ ")[-1])
        for i in range(len(ids)):
            qa_pairs = []
            
            if i == len(ids) - 1:
                break

            first = lines_dic[ids[i]].strip()  
            second = lines_dic[ids[i+1]].strip() 

            qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
            qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
            pairs.append(qa_pairs)
    
    return pairs

def train_tokenizer(pairs):
    if not os.path.exists('./data'):
        os.makedirs('./data')

    text_data = []
    file_count = 0

    for sample in tqdm.tqdm([x[0] for x in pairs]):
        text_data.append(sample)

        if len(text_data) == 10000:
            with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1

    paths = [str(x) for x in Path('./data').glob('**/*.txt')]

    # training custom tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train( 
        files=paths,
        vocab_size=30_000, 
        min_frequency=5,
        limit_alphabet=1000, 
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )

    if not os.path.exists('./bert-tokenizer'):
        os.makedirs('./bert-tokenizer')

    tokenizer.save_model('./bert-tokenizer', 'bert-it')
    tokenizer = BertTokenizer.from_pretrained('./bert-tokenizer/bert-it-vocab.txt', local_files_only=True)

    return tokenizer

