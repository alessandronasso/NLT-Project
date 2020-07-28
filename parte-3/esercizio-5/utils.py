import os, glob

def read_article(path):
    with open(path, 'r') as f:
        text = f.read()
    return text

def read_all_articles_in_folder(path):
    path = os.path.join(path, '*.txt')
    for file in glob.glob(path):
        yield read_article(file)
