import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import math
import datetime

train = pd.read_table('movie/train.tsv')
test = pd.read_table('movie/test.tsv')
sample = pd.read_csv('movie/sampleSubmission.csv')

document_bert = ["[CLS] " + str(s) + " [SEP]" for s in train.Phrase]
test_doc = ["[CLS] " + str(s) + " [SEP]" for s in test.Phrase]
# 예측 모델이기에 정답 없음. labels = test['Sentiment'].values


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

tokenized_texts = [tokenizer.tokenize(s) for s in document_bert]
tokenized_test = [tokenizer.tokenize(s) for s in test_doc]

MAX_LEN = 90
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_test]


for i in range(len(input_ids)): #zero padding
    add1 = list(0 for i in range(MAX_LEN-len(input_ids[i])))
    input_ids[i] = input_ids[i] + add1

for i in range(len(input_test)): #zero padding
    add2 = list(0 for i in range(MAX_LEN-len(input_test[i])))
    input_test[i] = input_test[i] + add2

#input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
attention_masks = []
attention_masks_test = []

for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

for seq in input_test:
    seq_mask2 = [float(i > 0) for i in seq]
    attention_masks_test.append(seq_mask2)


# 데이터 분리
train_inputs, validation_inputs, train_labels, validation_labels = \
train_test_split(input_ids, train['Sentiment'].values, random_state=42, test_size=0.2)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, #mask 분리
                                                       input_ids,
                                                       random_state=42,
                                                       test_size=0.2)
#to tensor
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
test_inputs = torch.tensor(input_test)
#test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks_test)

BATCH_SIZE = 88

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)


validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE*16)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=5)
model.cuda()

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8) #eps 0 division prevent
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def timesin(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


#training
# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    start = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {}'.format(step, len(train_dataloader), timesin(start)))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # Forward 수행
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # 로스 구함
        loss = outputs[0]

        # 총 로스 계산
        total_loss += loss.item()

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(timesin(start)))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # 시작 시간 설정
    start = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {}".format(timesin(start)))

print("")
print("Training complete!")

result = []
print("")
print("Test the data!")
start = time.time()
for step, batch in enumerate(test_dataloader):

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)

    # 배치에서 데이터 추출
    b_input_ids, b_input_mask = batch

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    print(" while processing... \nBatch {:>5,}  of  {:>5,}        Elapsed: {}".format(step, len(test_dataloader),
                                                                                      timesin(start)))
    out1 = torch.argmax(outputs[0], dim=1)
    result = result + out1.tolist()

print("  test took: {}".format(timesin(start)))

#result = torch.argmax(outputs[0], dim=1)
#print(result)
#print(len(result))

sample['Sentiment'] = result
sample.to_csv('sent.csv', index = False)


print("")
print("complete saving the result!")