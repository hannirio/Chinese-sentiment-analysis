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