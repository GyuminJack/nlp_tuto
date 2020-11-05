import os
import urllib
import pandas as pd
import tensorflow as tf
import logging
import numpy as np
import sentencepiece as spm
import sys
from models import *
from tensorboard.plugins.hparams import api as hp
from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def sentence_tokenize(input_df):
    with open("corpus.txt", "w") as f:
        for i in input_df.to_numpy():
            f.write(i)
            f.write("\n")
    parameter = '--input={} --model_prefix={} --vocab_size={} --model_type={}'
    input_file = 'corpus.txt'
    vocab_size = 32000
    prefix = 'spm_kor'
    model_type = 'bpe'
    cmd = parameter.format(input_file, prefix, vocab_size, model_type)
    spm.SentencePieceTrainer.Train(cmd)

def load_sentense_tokenize():
    sp = spm.SentencePieceProcessor()
    sp.Load('spm_kor.model') # prefix이름으로 저장된 모델
    return sp

def tokenize_padding(x, req_len):
    x = spm_tokenizer.EncodeAsIds(x)
    if len(x) >= req_len:
        return np.array(x[:req_len])
    elif len(x) < req_len:
        return np.array(x + ([0]*(req_len-len(x))))


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def loss_func(y_real, y_pred):
    return cross_entropy(y_real, y_pred)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

@tf.function
def train_step(train_x, train_y):
    with tf.GradientTape() as tape:
        predictions = model(train_x)
        loss = loss_func(train_y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(train_y, predictions)

@tf.function
def test_step(test_x, test_y):
    with tf.GradientTape() as tape:
        predictions = model(test_x)
        loss = loss_func(test_y, predictions)
    test_loss(loss)
    test_accuracy(test_y, predictions)


def main():
    global model
    global spm_tokenizer
    download_datas()
    train_x, train_y, test_x, test_y = get_datas()

    if os.path.isfile("spm_kor.model"):
        spm_tokenizer = load_sentense_tokenize()
    else:
        spm_tokenizer = sentence_tokenize(train_x)
        spm_tokenizer = load_sentense_tokenize()
    
    # model = BinaryModel()
    mirrored_strategy = tf.distribute.MirroredStrategy()
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])    
    with mirrored_strategy.scope():
        model = ConvModel()

    MAX_SEQ_LEN = 32
    BUFFER_SIZE = 10000
    BATCH_SIZE = 512
    print(MAX_SEQ_LEN, BATCH_SIZE)

    samples = train_x.apply(lambda x: tokenize_padding(x, MAX_SEQ_LEN)).to_numpy()
    samples = np.concatenate(samples).reshape(-1, MAX_SEQ_LEN)

    dataset = tf.data.Dataset.from_tensor_slices((samples, train_y.to_numpy().astype(np.float32).reshape(-1,1)))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    test_x = test_x.apply(lambda x: tokenize_padding(x, MAX_SEQ_LEN)).to_numpy()
    test_x = np.concatenate(test_x).reshape(-1,MAX_SEQ_LEN)
    test_y = test_y.to_numpy().astype(np.float32).reshape(-1,1)

    EPOCHS = 10000
    
    for epoch in range(EPOCHS):
        for train_x, train_y in dataset:
            train_step(train_x, train_y)

        template = '에포크: {}, 손실: {:.4f}, 정확도: {:.4f}, 테스트 손실: {:.4f}, 테스트 정확도: {:.4f}'
        test_step(test_x, test_y)

        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))
        
        if train_accuracy.result() > 0.9:
            model.save("sent_model")

if __name__ == "__main__":
    mirrored_strategy = tf.distribute.MirroredStrategy()
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    mode = "train"
    if mode == "train":
        main()
    else:
        my_sentence = input("Sentence : ")
        spm_tokenizer = load_sentense_tokenize()
        model = tf.keras.models.load_model("sent_model")
        # print(model(np.array([0]*20).reshape(-1,20)))
        MAX_SEQ_LEN = 30
        inputs = np.array(tokenize_padding(my_sentence, MAX_SEQ_LEN)).reshape(-1,MAX_SEQ_LEN)
        print(inputs)
        print(model(inputs))