import os
from os import listdir
from os.path import isfile, join
import numpy as np
#from tensorflow.contrib import learn
from collections import Counter
import tensorflow as tf
from tensorflow import keras

# This Data_Loader file is copied online
# data format pretrain 1,2,3,4,5,6
#finetune,
#input   1,2,3,4,5,6,,targetIDs
# output 1,2,3,4,5,6,'CLS',targetIDs  'CLS' denotes classifier

class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        # positive_examples = [[s[0],s[2:]]for s in positive_examples]

        # [user,itemseq] = [[s[0], s[2:]] for s in positive_examples]
        # print user
        colon=",,"
        source = [s.split(colon)[0] for s in positive_examples]
        target= [s.split(colon)[1] for s in positive_examples]


        max_document_length = max([len(x.split(",")) for x in source])
        # max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(source)
        x_array = tokenizer.texts_to_sequences(source) 
        for item in x_array:
            if len(item) < max_document_length:#小于指定的最大长度则用0填充 item.extend([0] * (max_document_length - len(item)))
                item.extend([0] * (max_document_length - len(item)))
        self.item = x_array
        self.item_dict = tokenizer.word_index

        max_document_length_target = max([len(x.split(",")) for x in target])
        vocab_processor_target = learn.preprocessing.VocabularyProcessor(max_document_length_target)
        self.target = np.array(list(vocab_processor_target.fit_transform(target)))  # pad 0 in the end
        self.target_dict = vocab_processor_target.vocabulary_._mapping

        # self.separator = len(self.item) + len(self.target)  # it is just used for separating such as :
        # self.separator = len(self.item_dict) # denote '[CLS]'
        self.separator = 0  # denote '[CLS]'
        lens = self.item.shape[0]
        # sep=np.full((lens, 1), self.separator)

        # self.example = np.hstack((self.item,sep,self.target))
        # concat source and one target

        self.example = np.array([])
        self.example = []
        for line in range(lens):
            source_line = self.item[line]
            target_line = self.target[line]
            target_num = len(target_line)


            for j in range(target_num):
                if target_line[j] != 0:
                    # np.array(target_line[j])
                    # unit = np.append(np.array(self.separator),source_line)
                    unit = np.append(source_line,  np.array(self.separator))
                    unit= np.append(unit,np.array(target_line[j]))
                    self.example.append(unit)

        self.example = np.array(self.example)









