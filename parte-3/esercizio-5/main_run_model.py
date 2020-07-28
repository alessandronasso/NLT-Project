import tensorflow as tf
import numpy as np
import dataset_loader
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = sys.argv[1]

tf.train.import_meta_graph(path + ".meta", clear_devices=True)

idx_to_category = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech'
}

MAX_SEQ_LEN = 350
EMBEDDING_SIZE = 300

def read_article_from_command_line(prompt):
    print(prompt)
    article = ''
    while True:
        par = input('')
        if par == '###END###':
            break
        article += par + '\n'
    return article

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, path)

    while(True):
        article = read_article_from_command_line("Write an article:\n")
        article = dataset_loader.preprocess_sentence(article, MAX_SEQ_LEN, EMBEDDING_SIZE)
        pred = sess.run("output/prediction_argmax:0", {"input_x:0": article})

        print()        
        print(idx_to_category[pred[0]])
        print()
