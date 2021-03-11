import re
import mistletoe
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English
from multiprocessing import Pool
from pathlib import Path
import csv

def preprocess(text):
    text = re.sub(r"\^", ' ', text)
    # For some reason the pushshift.io comments have HTML escapes in the markdown
    # we relpace them since the markdown parser doesn't recognie them.
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    # Render the markdown as an HTML string with mistletoe
    html = mistletoe.markdown(text)
    # Load HTML into BeautifulSoup 
    soup = BeautifulSoup(html, 'lxml')
    # Remove tables
    (t.extract() for t in soup('table'))
    # Remove code blocks
    (t.extract() for t in soup('code'))
    # Remove block quotes
    (t.extract() for t in soup('blockquote'))
    # Extract text
    text = soup.text
    # Replace whitespaces & other special chars with space 
    text = re.sub(r"\s|\||\*", ' ', text)
    # Normalize all sequences of more than 3 dots to 3 dots
    text = re.sub(r"\.\.+(?=\.)", '..', text)
    # Remove URLs
    text = re.sub(r"https?:\/\/[^\s]+", '', text)
    # Remove extra whitespace
    text = re.sub(r"^\s+|\s+$|\s+(?=\s)", '', text)
    return text

def preprocess_file(comment_file):
    nlp = English()
    spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)
    # Use the SpaCy tokenizer
    tokenized_comments = []
    print(comment_file)
    with open(comment_file, 'r') as f:
        reader = csv.DictReader(f)
        for comment in reader:
            try:
                comment_text = preprocess(comment['body'])
            except Exception as e:
                print("Preprocessing exception.")
                print(e)
                print(comment['body'])
                continue
            tokenized = [token.text for token in spacy_tokenizer(comment_text)]
            tokenized_comments.append({'comm': comment['subreddit'], 'text': comment_text})
    return tokenized_comments

corpus_dir = Path('reddit_sample')
with Pool(processes=12) as p:
    data = p.starmap(preprocess_file, [(f,) for f in corpus_dir.glob('*.csv')])

tokenized_corpus_dir = Path('reddit_tokenized')
output_files = {}
for shard in data:
    for comment in shard:
        if not comment['comm'] in output_files:
            output_files[comment['comm']] = open(tokenized_corpus_dir/f"{comment['comm']}.txt", 'w')
        output_files[comment['comm']].write(comment['text'] + '\n')

