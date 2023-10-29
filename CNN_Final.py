import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer

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
class CNNAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(CNNAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs) for fs in filter_sizes])
        self.attention = nn.Linear(num_filters * len(filter_sizes), 1)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids, attention_mask):
    embedded = self.embedding(input_ids).transpose(1, 2)
    conv_outputs = [torch.relu(conv(embedded)) for conv in self.convs]
    pooled_outputs = [torch.max_pool1d(output, output.shape[2]).squeeze(2) for output in conv_outputs]
    concatenated = torch.cat(pooled_outputs, dim=1)
    attention_weights = torch.tanh(self.attention(concatenated))
    attention_scores = torch.softmax(attention_weights, dim=1).transpose(1, 2)
    context_vector = torch.bmm(attention_scores, concatenated).squeeze(1)
    logits = self.classifier(context_vector)
    return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNAttentionModel(vocab_size=len(tokenizer.vocab), embedding_dim=128, num_filters=128, filter_sizes=[2, 3, 4], num_classes=2).to(device)
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
