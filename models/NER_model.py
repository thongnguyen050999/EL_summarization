import pandas as pd
import os
from itertools import chain
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model
from numpy.random import seed
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
import spacy
import pickle

seed(1)
tensorflow.random.set_seed(2)
nlp = spacy.load('en_core_web_lg')


## Existing bugs:
### Punctuation

class NER_data:

    def __init__(self, folder, text_file, entity_file, max_len, seg_len):
        self.folder = folder
        self.text_file = text_file
        self.entity_file = entity_file
        self.max_len = max_len
        self.seg_len = seg_len
        self.preprocess()

    def preprocess(self):
        text_df = pd.read_csv(os.path.join(self.folder, self.text_file))
        entity_df = pd.read_csv(os.path.join(self.folder, self.entity_file))

        sentences = {}
        texts = {}
        self.vocab = set()
        for i in range(len(text_df['text'].values)):
            print(i)
            if i >= 1000:
                break
            text = text_df['text'].values[i]
            text_id = text_df['text_id'].values[i]
            doc = nlp(text)
            words = [token.text for token in doc]
            sentences[text_id] = words
            texts[text_id] = text
            for word in words:
                self.vocab.add(word)

        positions = defaultdict(list)
        lengths = defaultdict(list)
        for id in sentences:
            print(id)
            entities = entity_df.loc[entity_df['text_id']
                                     == id]['entities'].values
            text = texts[id]
            sent = sentences[id]
            for ent in entities:
                try:
                    ent_tokens = []
                    ent_doc = nlp(ent)
                    for token in ent_doc:
                        ent_tokens.append(token.text)
                    pos = sent.index(ent_tokens[0])
                    positions[id].append(pos)
                    lengths[id].append(len(ent_tokens))
                except:
                    continue

        target_id = set()
        for id in sentences:
            if len(sentences[id]) <= self.max_len:
                target_id.add(id)


        X = defaultdict(list)
        y = defaultdict(list)
        for id in target_id:
            print(id)
            sent = sentences[id]
            for word in sent:
                X[id].append(tok2idx[word])
            label = np.zeros(len(sent))
            for i in range(len(positions[id])):
                pos = positions[id][i]
                ls = lengths[id][i]
                label[pos:pos+ls] = 1
            y[id] = label

        self.vocab.add('PAD')
        self.tok2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2tok = {idx: tok for idx, tok in enumerate(self.vocab)}

        true_X = []
        true_y = []

        for id in target_id:
            true_X.append(X[id])

            y_label = []
            for ele_y in y[id]:
                if not ele_y:
                    y_label.append([1, 0])
                else:
                    y_label.append([0, 1])
            true_y.append(y_label)

        pad_X = pad_sequences(true_X, maxlen=self.max_len, padding='post',
                              dtype='int32', value=len(self.vocab)-1)
        pad_y = pad_sequences(true_y, maxlen=self.max_len, padding='post',
                              dtype='int32', value=[1, 0])
        input_dim = len(self.vocab)
        input_length = self.max_len

        self.X = []
        self.y = []
        for i in range(pad_X.shape[0]):
            for j in range(0, self.max_len-self.seg_len, self.seg_len):
                self.X.append(pad_X[i][j:j+self.seg_len])
                self.y.append(pad_y[i][j:j+self.seg_len])

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.input_dim = input_dim
        self.input_length = input_length


class NER_model:

    def __init__(self, input_dim, input_length, output_dim, max_len, n_tags, path=None):
        self.max_len = max_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.n_tags = n_tags

        if path:
            self.path = path
        self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.input_dim,
                            output_dim=self.output_dim, input_length=self.input_length))
        model.add(Bidirectional(LSTM(units=self.output_dim, return_sequences=True), merge_mode='concat'))
        model.add(LSTM(units=self.output_dim, return_sequences=True))
        model.add(TimeDistributed(Dense(self.n_tags, activation='sigmoid')))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train(self, X, y, batch_size=32, validation_split=0.2, num_epochs=30):
        hist = self.model.fit(X, y, epochs=num_epochs, batch_size=batch_size,
                              validation_split=validation_split, verbose=1)
        self.model.save('./models/weights/ner_model.h5')

    def test(self):
        self.model.load_weights(self.path)


def main():
    folder = './dataset/KDWD'
    text_file = 'intro_text.csv'
    entity_file = 'intro_entity.csv'
    max_len = 256
    seg_len = 10
    output_dim = 32
    n_tags = 2
    mode = 'train'
    path = './models/weights/ner_model.h5'

    if mode == 'train':
        ner_data = NER_data(folder, text_file, entity_file, max_len, seg_len)
        pickleOut = open('./dataset/NER/ner_data.pkl', 'wb')
        pickle.dump(ner_data, pickleOut)
        pickleOut.close()

        ner_model = NER_model(ner_data.input_dim,
                              ner_data.seg_len, output_dim, max_len, n_tags)
        ner_model.train(ner_data.X, ner_data.y)
    else:
        pickleIn = open('./dataset/NER/ner_data.pkl', 'rb')
        ner_data = pickle.load(pickleIn)
        ner_model = NER_model(ner_data.input_dim,
                              ner_data.seg_len, output_dim, max_len, n_tags, path)
        ner_model.test()

        N = 100
        labels = ner_model.model.predict(ner_data.X)
        tmp_labels = labels[:N]
        tmp_datum = ner_data.X[:N]

        for i in range(N):
            print(i)
            tmp_label = tmp_labels[i]
            tmp_data = tmp_datum[i]
            tmp_y = ner_data.y[i]

            for ele in tmp_data:
                print(ner_data.idx2tok[ele], end=' ')
            print()
            print('Predicted:',end=' ')
            for ele in tmp_label:
                if ele[1] >= 0.4:
                    print(1, end=' ')
                else:
                    print(0, end=' ')
            print()
            print('Label:',end=' ')
            for ele in tmp_y:
                if ele[1] >= 0.4:
                    print(1, end=' ')
                else:
                    print(0, end=' ')
            print()
            print('='*100)


if __name__ == '__main__':
    main()
