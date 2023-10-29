import pandas as pd
import numpy as np

# 读取文件
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ChnSentiCorp_htl_all.csv', encoding='utf-8')

# 清洗数据
df = df.dropna() # 删除空值
df = df[df['review'].str.len() > 0] # 删除内容为空的行

# 筛选数据
df = df[['review', 'label']]

# 分成训练集和测试集
train_data = df.sample(frac=0.8, random_state=0)
test_data = df.drop(train_data.index)

# 保存数据
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

!pip install transformers
import torch
from transformers import BertTokenizer

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 定义文本分词函数
def tokenize(text):
    return tokenizer.tokenize(text)

# 加载训练数据
train_data = pd.read_csv('train_data.csv')

# 分词
train_data['review'] = train_data['review'].apply(tokenize)

# 保存分词后的数据
train_data.to_csv('train_data_tokenized.csv', index=False