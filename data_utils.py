import numpy as np
import tensorflow as tf
import json
from collections import Counter
from memory_profiler import profile
max_document_len = 40

def dump_list_to_dict(filename):
    with open(filename,"r") as f:
        rows = [(row,i) for i, row in enumerate(f.readlines())]
    dic = dict(rows)
    with open("data/words.json","w") as f:
        json.dump(dic,f)

#@profile
def build_word_list(filename):
    c = Counter()
    rows = []
    with open(filename, "r") as f:
        for i,row in enumerate(f.readlines()):
            if i%10000==0 and i!=0: 
                print(i)
                c.update(Counter(rows))
                rows = []
            row = row.strip().split()
            rows+=row
        if len(rows)!=0:
            c.update(Counter(rows))
    return c

def build_word_dict(filename,start_index=4):
    words = []
    with open(filename, "r") as f:
        for row in f.readlines():
            row = row.strip().split()
            #print(row)
            words+=list(set(row))
        #words = f.read().replace("\n", "").split()
    words = set(words)
    word_dict = dict(((v,i+start_index) for i, v in enumerate(words)))
    del words
    gc.collect()
    return word_dict

def build_dataset(filename, word_dict):
    with open(filename, "r") as f:
        lines = f.readlines()
        data = map(lambda s: s.strip().split()[:max_document_len], lines)
        data = map(lambda s: ["<bos>"] + s + ["<eos>"], data)
        data = map(lambda s: [word_dict.get(w, word_dict["<unk>"]) for w in s], data)
        data = map(lambda d: d + (max_document_len +2 - len(d)) * [word_dict["<pad>"]], data)

    return data

def get_dataset(filename):
    with open(filename, "r") as f: 
        rows = [row.strip().split() for row in f.readlines()]
    return rows

def get_word_dict(filename):
    with open(filename, "r") as f:
        word_dict = json.load(f)
    return word_dict

def batch_iter(inputs, batch_size, num_epochs):
    #inputs = np.array(inputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield np.array(inputs[start_index:end_index])

def data_loader(filename,batch_size=64,epochs=5,shuffle=True):
    dataset = tf.data.TextLineDataset([filename])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element 
