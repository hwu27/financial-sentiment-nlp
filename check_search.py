import search
import transformers
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import pandas as pd
import numpy as np
import math

# BERT class for the model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased') # pre-trained BERT model
        self.l2 = torch.nn.Dropout(0.3) # dropout layer for regularization
        self.l3 = torch.nn.Linear(768, 3) # output layer with 3 classes
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

model = BERTClass()
device = 'cuda' if cuda.is_available() else 'cpu'
model.to(device)
model.load_state_dict(torch.load('bert_model.pth'))

visited_titles = search.search()
df = pd.DataFrame({'text': visited_titles})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 512
data = CustomDataset(df, tokenizer, max_len)

BATCH_SIZE = 4
data_loader = DataLoader(data, batch_size=BATCH_SIZE)

predictions = []
for batch in data_loader:
    ids = batch['ids'].to(device, dtype = torch.long)
    mask = batch['mask'].to(device, dtype = torch.long)
    token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
    outputs = model(ids, mask, token_type_ids)
    predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

predictions = (np.array(predictions) >= 0.5).astype(int)

negative_count = 0
neutral_count = 0
positive_count = 0
for i in range(len(predictions)):
    if predictions[i][0] == 1:
        negative_count += 1
    elif predictions[i][1] == 1:
        neutral_count += 1
    elif predictions[i][2] == 1:
        positive_count += 1
    else:
        neutral_count += 1 # sometimes the model is unsure, so we default to neutral
print(predictions)

confidence_rating = 0

diff_count = positive_count - negative_count
confidence_rating = math.floor(diff_count / len(predictions) * 100)

if (confidence_rating == 0):
    print("The overall sentiment of the financial news is neutral.")
if (confidence_rating > 0):
    print("The overall sentiment of the financial news is positive with a positive confidence rating of " + str(confidence_rating) + "%.")
if (confidence_rating < 0):
    print("The overall sentiment of the financial news is negative with a negative confidence rating of " + str(abs(confidence_rating)) + "%.")
