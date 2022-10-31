import os
from os import listdir
from os.path import isfile, join
import numpy as np
#from tensorflow.contrib import learn
from collections import Counter
import tensorflow as tf
from tensorflow import keras

# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        max_document_length = max([len(x.split(",")) for x in positive_examples])
        #max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(positive_examples)
        x_array = tokenizer.texts_to_sequences(positive_examples) 
        for item in x_array:
            if len(item) < max_document_length:#小于指定的最大长度则用0填充 item.extend([0] * (max_document_length - len(item)))
                item.extend([0] * (max_document_length - len(item)))
        self.item = x_array
        self.item_dict = tokenizer.word_index



        # added to calculate word frequency
        # allitems_hassamewords=list()
        # for line in self.item:
        #     for ele in line:
        #         allitems_hassamewords.append(ele)
        #
        # counts = Counter(allitems_hassamewords)
        # most_com=counts.most_common(10)
        # print allitems_hassamewords




