import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
import spacy
import re
from spacy.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import time, math
import itertools

train = pd.read_table('movie/train.tsv')
test = pd.read_table('movie/test.tsv')

sample = pd.read_csv('movie/sampleSubmission.csv')

print(train['Sentiment'].value_counts()) # 분포 확인.
print(sample.head())
all_data = pd.concat([train, test]) # for tokenizer concat data.


#all_data['Phrase'] = all_data['Phrase'].str.lower()
#all_data['Phrase'] = all_data['Phrase'].str.replace(r"([.,!?])", r" |1 ")
#all_data['Phrase'] = all_data['Phrase'].str.replace(r"[^a-zA-Z.,!?]+", r" ")

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)
tokens = []

# 표제어 추출.
for sentence in tokenizer.pipe(all_data['Phrase']):
    doc_tokens = []
    for token in sentence:
        token.lemma_
        if (token.is_stop == False) & (token.is_punct == False):
            doc_tokens.append(token)
    tokens.append(doc_tokens)


token_1 = list(itertools.chain(*tokens))
token_set = set(token_1)
wordset = {tkn: i+2 for i, tkn in enumerate(token_set)}
wordset['<unk>'] = 0
wordset['<pad>'] = 1

class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self._vectorizer = vectorizer

    def get_vectorizer(self):
        return self._vectorizer
    def __len__(self):
        return self._target_size
    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        review_vector = \
            self._vectorizer.vectorize(row.review)
        rating_index = \
            self._vectorizer.rating_vocab.lookup_token(row.review)
        return {'x_data' : review_vector
                'y_target' : rating_index}

''' 왜 안될까
for sentence in all_data['Phrase']:
    
    doc_tokens = [re.sub(r"[^a-z0-9]", "", token.lower()) for token in sentence]
    
        doc_tokens.append(token.text.lower())
    tokens2.append(nlp(doc_tokens))
'''

#all_data['tokens'] = tokens
#print(all_data['tokens'].head())
#print(len(tokens))

#tfidf_tuned = TfidfVectorizer(stop_words = 'english', max_features=15)

#dtm_tfidf_tuned = tfidf_tuned.fit_transform(all_data['Phrase'])
#dtm_tfidf_tuned = pd.DataFrame(dtm_tfidf_tuned.todense(), columns=tfidf_tuned.get_feature_names_out())
#print(dtm_tfidf_tuned.shape)
#all_data['token'] = dtm_tfidf_tuned

train2 = tokens[:len(train)]
test2 = tokens[len(train):]

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(train2, train['Sentiment'], test_size=0.2)

print(np.array(train_x))
train_x = torch.from_numpy(np.array(train_x))
train_y = torch.tensor(train_y)
val_x = torch.tensor(val_x)
val_y = torch.tensor(val_y)

dataset = TensorDataset(train_x, train_y)
testset = TensorDataset(val_x, val_y)
dataloader = DataLoader(dataset, batch_size= 20, shuffle= True)
testloader = DataLoader(testset, batch_size= 100, shuffle= False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE = {}".format(DEVICE))
class model(nn.Module):
    def __init__(self, embedding_dim, context_size):
        super().__init__() # super().__init__
        self.embedding = nn.Embedding(num_embeddings=len(wordset), embedding_dim=embedding_dim)
        self.layer1 = nn.Linear(context_size*embedding_dim, 128)
        self.layer2 = nn.Linear(128, len(wordset))
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x): # 28*28 영상이 64개! 병렬성 증가.
        embeds = self.embedding(x).view((1,-1)).to(DEVICE)
        out = self.relu(self.layer1(embeds))
        out = self.layer2(out)
        #out = self.softmax(out, dim=1)

        return out


def timesin(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


model = model(3, 2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 요즘엔 adam을 많이 쓰기도 함.
loss_stack = []

start = time.time()
for epoch in range(10):
    running_loss = 0
    for i, data in enumerate(dataloader):
        x, y = data
        optimizer.zero_grad()
        outputs = model.forward(x.to(DEVICE))
        loss = criterion(outputs, y.to(DEVICE))
        loss.backward()
        optimizer.step()
        running_loss += loss
    if epoch % 100 == 0:
        print('[Epoch %d, step = %5d] loss:%.3f, time = ' % (epoch + 1, i + 1, running_loss), timesin(start))
        loss_stack.append(running_loss)
        running_loss = 0.0
print(timesin(start))