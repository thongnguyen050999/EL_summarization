from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, concatenate, Bidirectional, LSTM, Dense, Dropout
import pandas as pd
import os
from commands.get_candidates import preprocess_text, get_forest, predict
import spacy
import pickle
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
import re


class EL_model:

    def __init__(self, data, text_input_shape, graph_input_shape, embed_dim):
        self.text_input_shape = text_input_shape
        self.graph_input_shape = graph_input_shape
        self.dict_size = len(data.embedding_dict)
        self.embed_dim = embed_dim
        self.data = data
        self.build_KG_context_model()

    def build_KG_context_model(self):
        inp_0 = Input(shape=(self.text_input_shape, self.embed_dim,))
        inp_1 = Input(shape=(self.graph_input_shape,))
        out_0 = Bidirectional(LSTM(units=128, return_sequences=False))(inp_0)
        w = concatenate([out_0, inp_1])
        w = Dense(256, activation='relu')(w)
        w = Dropout(0.2)(w)
        out = Dense(2, activation='sigmoid')(w)
        model = Model(inputs=[inp_0, inp_1], outputs=out)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train(self, epochs=10):
        self.model.fit([self.data.lstm_train, self.data.graph_train],
                       self.data.y_train, epochs=epochs)


class EL_data:

    def __init__(self, folder, text_file, entity_df_file, kdwd_entity_file, ent_embedding_file, rel_embedding_file, forest_file, glove_file, window_size=10, permutations=128, num_results=10):
        self.window_size = window_size

        pickleIn = open(forest_file, 'rb')
        self.forest = pickle.load(pickleIn)
        self.folder = folder
        self.kdwd_entity_file = kdwd_entity_file
        self.text_file = text_file
        self.entity_df_file = entity_df_file
        self.glove_file = glove_file
        self.permutations = permutations
        self.num_results = num_results
        self.ent_embedding_file = ent_embedding_file
        self.rel_embedding_file = rel_embedding_file
        self.load_data()

    def load_data(self):
        with open(self.ent_embedding_file, 'rb') as f:
            self.ent_embeddings = pickle.load(f)

        with open(self.rel_embedding_file, 'rb') as f:
            self.rel_embeddings = pickle.load(f)

        self.text_df = pd.read_csv(os.path.join(self.folder, self.text_file))
        self.entity_df = pd.read_csv(
            os.path.join(self.folder, self.entity_df_file))
        self.get_kdwd_entities()
        self.nlp = spacy.load('en_core_web_lg')

        self.embedding_dict = {}
        with open(self.glove_file, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], 'float32')
                self.embedding_dict[word] = vector

        self.prepare_data()

    def preprocess_text(self,text):
        text=re.sub(r'[^\w\s]','',text)
        tokens=text.lower()
        tokens=re.split(r'[\s_]',tokens)
        return '_'.join(tokens)

    def get_context(self, tokens, start, end):
        context = tokens[max(0, start-self.window_size):start] + \
            tokens[end:end+self.window_size]
        return context

    def prepare_data(self):
        textID2entity = defaultdict(list)
        text_id_set = set()
        for i in range(self.entity_df.shape[0]):
            print(i)
            text_id = self.entity_df.iloc[i]['text_id']
            textID2entity[text_id].append(
                self.preprocess_text(self.entity_df.iloc[i]['entities']))
            text_id_set.add(text_id)
            if i >= 10000:
                break

        data = []
        for id in text_id_set:
            print(id)
            text = self.text_df.loc[self.text_df['text_id']
                                    == id]['text'].values[0]

            tokens = []
            doc = self.nlp(text)
            for token in doc:
                tokens.append(token)

            entities = []
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    continue
                entities.append([ent.text, ent.start, ent.end])

            target_entities = textID2entity[id]

            for entity in entities:
                ent_text = entity[0]
                results = predict(ent_text, self.kdwd_entities,
                                  self.permutations, self.num_results, self.forest)
                batch = {}
                batch['negative'] = []
                if results:
                    for res in results:
                        if res in target_entities:
                            context = self.get_context(
                                tokens, entity[1], entity[2])
                            batch['pos'] = res
                            batch['context'] = context
                        else:
                            batch['negative'].append(res)
                if 'context' in batch:
                    data.append(batch)

        self.data = data
        self.preprocess_data()

    def get_kdwd_entities(self):
        kdwd_entities = []
        with open(os.path.join(self.folder, self.kdwd_entity_file), 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                ent, id = line.strip().split('\t')
                kdwd_entities.append(ent)
        self.kdwd_entities = kdwd_entities

    def preprocess_data(self):
        lemmatizer = WordNetLemmatizer()
        X = []
        for ele in self.data:
            embeds = []
            text = ele['context']
            for word in text:
                try:
                    token = word.text.lower()
                    token = lemmatizer.lemmatize(token)
                    embed = self.embedding_dict[token]
                    embeds.append(embed)
                except:
                    embed = np.random.random(50)
                    embeds.append(embed)

            if len(embeds) < 2*self.window_size:
                for i in range(2*self.window_size-len(embeds)):
                    embeds.append(np.zeros(50))
            embeds = np.array(embeds)

            pos_ent = ele['pos']
            pos_idx = self.kdwd_entities.index(pos_ent)
            pos_ent_embed = self.ent_embeddings(torch.LongTensor([pos_idx]))
            X.append((embeds, pos_ent_embed, 1))

            for neg_ent in ele['negative']:
                neg_idx = self.kdwd_entities.index(neg_ent)
                neg_ent_embed = self.ent_embeddings(
                    torch.LongTensor([neg_idx]))
                X.append((embeds, neg_ent_embed, 0))

        lstm_train = []
        graph_train = []
        y_train = []

        for x in X:
            lstm_train.append(x[0])
            graph_train.append(x[1].squeeze(0).detach().numpy())
            if x[2]:
                y_train.append([0, 1])
            else:
                y_train.append([1, 0])

        self.lstm_train = np.array(lstm_train)
        self.graph_train = np.array(graph_train)
        self.y_train = np.array(y_train)


def main():
    folder = './dataset/KDWD'
    text_file = 'intro_text.csv'
    entity_df_file = 'intro_entity.csv'
    kdwd_entity_file = 'training_data//entity2id.txt'
    forest_file = './models/weights/lsh_forest.pkl'
    glove_file = './models/weights/glove/glove.6B.50d.txt'
    ent_embedding_file = './models/OpenKE/ent_embeddings.pkl'
    rel_embedding_file = './models/OpenKE/rel_embeddings.pkl'

    el_data = EL_data(folder, text_file, entity_df_file,
                      kdwd_entity_file, ent_embedding_file, rel_embedding_file, forest_file, glove_file)

    text_input_shape = 20
    graph_input_shape = 200
    embed_dim = 50

    el_model = EL_model(el_data, text_input_shape,
                        graph_input_shape, embed_dim)
    el_model.train()


if __name__ == '__main__':
    main()
