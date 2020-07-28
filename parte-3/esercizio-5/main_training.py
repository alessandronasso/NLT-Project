from embeddings import *
import dataset_loader
import random
import config_helper
import tensorflow as tf
from cnn_model import NLPCNN
import time
import numpy as np

config = config_helper.load_config("cnn_config.json")
cnn_config = config["cnn_config"]
training_config = config["training_config"]
other_config = config["other_config"]

maxSeqLen = cnn_config["max_sequence_length"]
dropout_keep = cnn_config["dropout_keep_prob"]
numDimensions = cnn_config["num_dimensions"]
iterations = training_config["iterations"]
batchSize = training_config["batch_size"]
max_grad_norm = training_config["max_grad_norm"]
train_set_ratio = training_config["train_set_ratio"]
dataset_path = training_config["dataset_path"]


def evaluate(x, y, batch_size):
    tot_accuracy = 0
    tot_loss = 0
    for i in range(len(x) // batch_size):
        nextBatch = x[i * batch_size: batch_size + i * batch_size]
        nextBatchLabels = y[i * batch_size: batch_size + i * batch_size]
        feed_dict = {
            model.input_x: nextBatch,
            model.input_y: nextBatchLabels
        }
        acc, loss = sess.run([model.correct_count, model.loss], feed_dict)
        tot_accuracy += acc
        tot_loss += loss
    
    remaining = len(x) % batch_size

    if remaining != 0:
        feed_dict = {
            model.input_x: x[len(x) - remaining: len(x)],
            model.input_y: y[len(x) - remaining: len(x)]
        }
        acc, loss = sess.run([model.correct_count, model.loss], feed_dict)
        tot_accuracy += acc
        tot_loss += loss

    tot_accuracy = tot_accuracy / len(x)
    div = (len(x) // batchSize) if remaining == 0 else (len(x) // batchSize) + 1
    tot_loss = tot_loss / div
    return tot_accuracy, tot_loss

if __name__ == '__main__':

    # Loading and preprocessing dataset
    category_idx = dataset_loader.load_categories(dataset_path)
    X, y = dataset_loader.randomize_dataset(*dataset_loader.load_dataset(dataset_path, category_idx))
    X = dataset_loader.pad(X, maxSeqLen, numDimensions)
    split_index = round(len(y) * train_set_ratio)
    X, testX = X[:split_index], X[split_index:]
    y, testY = y[:split_index], y[split_index:]

    tf.reset_default_graph()

    model = NLPCNN(
        maxSeqLen,
        len(category_idx),
        numDimensions,
        cnn_config["filter_sizes"],
        cnn_config["num_filters"],
        cnn_config["l2_lambda"],
        training_config["learning_rate"],
        training_config["train_on_gpu"],
        training_config["max_grad_norm"],
        noise = training_config["input_noise"]
    )

    # TRAINING
    best_accuracy = 0
    with tf.Session() as sess:
        log_path = other_config["logdir"] + "/run" + str(time.time())
        model_save_path = log_path + "/model"
        log_path = log_path + "/log"
        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        saver = tf.train.Saver(max_to_keep=1)

        sess.run(tf.global_variables_initializer())

        print("Training started")

        fetches = {
            "loss": model.loss_summary,
            "global_step": model.global_step,
            "train_op": model.train_op
        }

        fetches_wb_summ = {
            "out_w": model.output_w_summary,
            "out_b": model.output_b_summary
        }

        for i, summary in enumerate(model.conv_w_summaries):
            fetches_wb_summ["conv_w_%s" % i] = summary
        for i, summary in enumerate(model.conv_b_summaries):
            fetches_wb_summ["conv_b_%s" % i] = summary

        for i in range(iterations):
            for j in range(len(X) // batchSize):
                nextBatch = X[j * batchSize: batchSize + j * batchSize]
                nextBatchLabels = y[j * batchSize: batchSize + j * batchSize]
                result = sess.run(fetches, {model.input_x: nextBatch, model.input_y: nextBatchLabels, model.dropout_keep_prob: dropout_keep, model.is_training: True})
                train_writer.add_summary(result["loss"], result["global_step"])

                remaining = len(X) % batchSize

                if remaining != 0:
                    feed_dict = {
                        model.input_x: X[len(X) - remaining: len(X)],
                        model.input_y: y[len(X) - remaining: len(X)],
                        model.dropout_keep_prob: dropout_keep,
                        model.is_training: True
                    }
                    
                    result = sess.run(fetches, feed_dict)
                    train_writer.add_summary(result["loss"], result["global_step"])

            tot_accuracy, tot_loss = evaluate(testX, testY, batchSize)
            acc_summ, loss_sum = sess.run([model.acc_summary, model.test_loss_summary], {model.tf_accuracy_ph: tot_accuracy, model.tf_test_loss_ph: tot_loss})
            train_writer.add_summary(acc_summ, result["global_step"])
            train_writer.add_summary(loss_sum, result["global_step"])

            if tot_accuracy > best_accuracy:
                best_accuracy = tot_accuracy
                saver.save(sess, model_save_path + "/model.ckpt", global_step=result["global_step"], write_meta_graph=True)

            tot_train_accuracy, _ = evaluate(X, y, batchSize)
            train_acc_sum = sess.run(model.train_acc_summary, {model.tf_train_accuracy_ph: tot_train_accuracy})
            train_writer.add_summary(train_acc_sum, result["global_step"])

            result_wb = sess.run(fetches_wb_summ, {})
            for k, v in result_wb.items():
                train_writer.add_summary(v, result["global_step"])

        print("Final Test Accuracy: " + str(tot_accuracy))
        print("Best Test Accuracy: " + str(best_accuracy))
