!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import torch_geometric.nn as geom_nn

# Data loading and preprocessing
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/simplifyweibo_4_moods.csv', encoding='utf-8')
df = df.dropna()
df = df[df['review'].str.len() > 0]
df = df[['review', 'label']]

train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def tokenize(text):
    return tokenizer.encode_plus(text, padding='max_length', max_length=256, truncation=True, return_tensors='pt')

train_data['review'] = train_data['review'].apply(tokenize)
test_data['review'] = test_data['review'].apply(tokenize)

# Dataset and DataLoader
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_ids = item['review']['input_ids'].squeeze(0)
        attention_mask = item['review']['attention_mask'].squeeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)
        return input_ids, attention_mask, label

train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Definition
class AttentionMixtureModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(AttentionMixtureModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.attention = nn.Linear(hidden_size, 1)
        self.gcn = geom_nn.GCNConv(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = bert_output.last_hidden_state

        # 构建一个完全连接的图
        edge_index = torch.tensor([[i, j] for i in range(hidden_states.size(1)) for j in range(hidden_states.size(1))], dtype=torch.long).t().contiguous().to(hidden_states.device)
        gcn_output = self.gcn(hidden_states, edge_index)

        attention_scores = torch.softmax(self.attention(gcn_output), dim=1)
        context_vector = torch.sum(attention_scores * gcn_output, dim=1)
        logits = self.classifier(context_vector)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionMixtureModel(hidden_size=768, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Performance Metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

