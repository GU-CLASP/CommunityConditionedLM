import re
import mistletoe
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English
from multiprocessing import Pool
from pathlib import Path
import csv

def preprocess(text)
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

def preprocess_file(comment_file):
    nlp = English()
    spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)
    # Use the SpaCy tokenizer
    tokenized_comments = []
    print(comment_file)
    with open(comment_file, 'r') as f:
        reader = csv.DictReader(f)
        for comment in reader:
            comment_text = preprocess(comment['body'])
            tokenized = [token.text for token in spacy_tokenizer(comment_text)]
            tokenized_comments.append((comment['subreddit'], comment_text))
    return tokenized_comments

corpus_dir = Path('reddit_sample')
with Pool(processes=8) as p:
    p.starmap(preprocess_file, corpus_dir.glob('*.csv'))
