---
layout: post
title: PyTorch 与深度学习
published: true
---


## PyTorch 运算

张量运算
- 注意先从直觉直观的角度设想，然后再选择优化的实现方案。
  - 通常第一维度是 batch，最后维是特征。即每行一组数据。
- 创建一个 `x = torch.rand(5, 3)` 非常适合用来调试。
- 矩阵运算，注意存在 batch 维。
	- a + b 向量相加，维度相等（或者其中一个是标量）。
	- input_gru = torch.cat((embedded, context), dim=2) 两个特征合并。
	- 乘法 `torch.matmul()` 或 `@` 。针对2维矩阵的运算，对最低的两个维度，需要 batch 维一致。
- 改变形状
	- `.view` `.reshape(-1)` 注意行优先，可以用来 flatten 或逆运算。
	  - `inputs.unsqueeze(0)` 和 `.squeeze()` 增加一个维度，用来增加 batch 或者 length 为 1。很有用的简写。需要时可以再复制，或者用广播。比如说测试的时候单个样本需要调用模型的时候很有用。 
	- `.repeat()` 或 `.expand()` 用来复制。
	- 转置 `transpose()` 或 `permute() ` 方法可用于交换张量的两个或多个维度
- 数组
	-  `.argmax(dim=-1)` 得到分类编号，这时候不需要 `.softmax(-1)`。
	-  求和 `.sum` 求平均 `.mean`，可以指定维度，比如对特征，也可以得到总的标量。
- 注意
  - 很多时候第一维是 batch，基本运算需要显式地考虑，模型运算则隐含。
  - 对于图片要考虑通道数，长宽。对于文本要考虑长度。


模型
- 模块 Module 包含 batch 维（隐含的，调用者不需要考虑），通常有可学习参数，也可以当张量运算使用（如果没有可学习参数的话）。
- 层 Layer，是模块的组成部分，自身也是一个模块。包括 batch 维，模块都隐含 batch 维，不同于基础的矩阵运算。
	- Linear(28 * 28, 128), 输入输出的大小。
	- ReLU() 放在 Linear 之后
	- Sequential(Linear(100, 64), ReLU(), Linear(64, 10))
	- CNN 用于图像，也用于并行的场景。
    - RNN、Transformer 用于序列。
- 模型 Model，完整的模型也可以作为一个层来调用，也可以直接用来完成任务。
	- 注意输入输出格式，和约定。数组格式。可能需要转换。可能有多个输入，可能输入是个元组或者字典。
		- 调用第三模型的时候根据文档和示例中描述。
		- 因为模型的输入输出都是张量，图片和文本都需要预处理。模型和预处理模块是一一对应的，必须匹配。
		- 分类任务往往输出一个数组，需要转换为编号，字符序列需要转换为文本。
    - 表示层的模型用来提取特征，往往经过预训练了。
- 图像和文本模型
	- 图像
		- cnn 可以用于提取特征。
        - 用 1*1 卷积核可以代替全连接层。
        - 图像分类，特征提取 模型 `models.resnet18()` 可以使用或者参考一下 pytorch 自带的实现。
		- 目标检测模型
            - `models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")`
            - Yolo, SSD
	- 文本
		- embedding 用于词表转词向量，预处理要分词转编号。查表法，可学习。
		- `LSTM(input_dim, hidden_dim, batch_first=True)` 输入是 序列x词向量。内部会对输入的每个词依次循环调用。
		- `BertForSequenceClassification.from_pretrained("bert-base-uncased")` 想获得上下文相关的词向量的话可以直接调用。
- 预处理，也是整个模型不可替换的一部分，接受原始输入。
- 输出是向量或者标量，有时候需要再次转化。

推断
- 调用模型时，需要先转化输入，最后再转化输出。
- 可以用肉眼观察看卡效果，也可以用评价指标判断模型。
- 推断模式不需要计算梯度。

训练
- 损失函数，也是一种层
	- mse_loss 用于拟合任务
		- `loss = self.model(batch).sum()`
	- CrossEntropyLoss() 用于多分类任务，注意模型最后不需要激活函数，不需要 softmax，已经隐含在损失函数内了。
	- 二分类也可以当作多分类处理。
- 优化器
	- `optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)`
	- `optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)`
- 评估
	- `preds = self.model(imgs).argmax(dim=-1)` 从向量得到编号。
	- `acc += (preds == labels).sum().item()` 得到正确的个数
	- 机器翻译的评价指标 BLUE。
	- 评价指标和损失函数不一定是同一个。因为实现差异损失函数的具体大小没有实际意义。
	- 评估首先是肉眼看一下结果，甚至不一定是数字指标。
- 训练技巧
	- 学习率调整
	- dropout nn.Dropout(dropout) 位置在线性层前。
	- 训练过程中打印损失函数变化，画出折线图。
    - 用小样本做实验，看模型能否运行，是否有初始的效果。
	- torch.nn.utils.clip_grad_norm_(...) 防止梯度爆炸
    - 训练过程中保存模型，或者保存中间结果。
- 其他提升方法
    - 归一化
	  -  nn.BatchNorm2d(c_out), 位置在线性层后。
	  -  层归一化。 nn.LayerNorm(self.hparams.model_dim),  位置在线性层后。
- 数据集
	- DataSet 相当于一个数组。
	- DataLoader 用来分 batch。指定 batch size。序列可能是填 0。
	- 可能需要预处理或者增强。图像 transform（翻转、缩放），文本 tokenize（词典转编号）。在 dataloader 之前可以提升性能。
	- 划分训练、测试/验证。
	- 文本的输入是字符序列，分词，根据词表转编号，然后 embedding ，最终变成向量。
	- 见后面的代码，利用 pytorch 或其他库提供的工具。
- 备注
	- 通常实现为命令行模式方便调用。
	- 训练过程大多数情况不需要手工调整。
	- 首先找一个完整的代码跑一下。
	- 动态图很方便交互调试，用固定的输入来尝试调用，看对应输出的格式。尽量各维度长度不一致，及时报错。
	- 对于整个过程只关心必要的部分。
	- 对模块拼接，不要从零开始，利用或者修改已有的能测试的代码。
	- 调用第三方模块的时候，看一下输入输出格式，以及约定。这个不固定，特别是有细节差异。含预训练权重。

任务
- 常见任务
	- 文本。分类。总结。
		- 文本分类 数据集 AG NEWS 新闻分类，IMDB 评论 情感分类 模型 rnn/lstm，fasttext, bert, 
		- 序列生成，翻译。rnn/lstm, bert, gpt 
	- 图像。分类。检测。
		- 图像分类 MNIST 手写数字识别 模型 cnn, resnet, 
		- 目标检测 模型 yolo
	- 传统模型
		- 文本分类 tfidf+svm
		- 
- 发展历史
	- 图像分类 cnn, vgg, resnet,
	- 文本
		- 词向量
		- 序列模型 rnn, lstm, attention, bert, gpt, 
- 代码参考
	- https://lightning.ai/docs/pytorch/stable/tutorials.html
	- [Annotated Research Paper Implementations](https://nn.labml.ai/)
	- [PyTorch Tutorials](https://pytorch.org/tutorials/)
	- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


其他补充
- img = img.unsqueeze(0) 用来添加 batch
- topk 用来得到最大的下边编号。
- 特征层可以固定权重以减少微调开销。
- 图像输入注意格式，尺寸，通道，归一化。
- 用 imshow 的时候要根据预处理逆转换回去。
- model_ft = models.resnet18(weights='IMAGENET1K_V1')
- bert 可以句子分类，可以词分类，可以序列生成。




## 学习与研究

- 论文
	- [Attention Is All You Need](https://ar5iv.labs.arxiv.org/html/1706.03762)
	- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://ar5iv.labs.arxiv.org/html/1810.04805)
- chatgpt
	- 调用工具
	- 通过 api 使用，输入和返回是文本。分类任务可以返回标签字符串。 
	- 利用 示例和微调 都可以提升具体任务的效果。根据数据量选择不同的方案。
- 其他机器学习库
	- sklearn 传统的机器学习
	- waka
	- pytorch
	- keras 基于 tf 已淘汰，
	- hugging face
	- 图模型
	- 词向量，主题模型 LDA。
	- numpy  提供基本的矩阵运算
	- scipy 有最优化工具箱。
- 文本上的算法。
	- 文本分类
- 教材
	- 李航。
	- 文本上的算法。
	- hulu 葫芦书（两本）有点老，但是有一些有趣的细节问题。答案仅参考，不一定好。
- 论文
	- 问题是什么，和基本的问题有什么区别，评估方法是什么。
	- 传统方法是什么，相似问题上的方法是是什么，在哪个基础方法上做出哪些改进。
	- 如何评估，有什么实际应用，分析各个组件的作用和场景。
	- 一些 survey。
- 相关会议与杂志。


## 模块


### 层/模块


一个简单的双层感知机

```python
import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model = MLP()
print(model)
y = torch.rand(2, 28*28)
print(model(y))
```

pytorch-lighting 也一样
```python
class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
		self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
		self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

	def forward(self, batch, batch_idx):
		x, _ = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log("train_loss", loss)
		return loss
```

### CNN


改进
VGG 用 3x3 卷积核堆叠。
```python
conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
nn.MaxPool2d(kernel_size=2, stride=2)
```

resnet


### RNN
### Attention
- Q,K,V 是 $batch \times length \times dim$ 
- $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- self attention 即 Q=K=V 用于 编码
- decoder 可以用来给输入添加上下文。

```python
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

seq_len, d_k = 3, 2
pl.seed_everything(42)
q = torch.randn(seq_len, d_k)
k = torch.randn(seq_len, d_k)
v = torch.randn(seq_len, d_k)
values, attention = scaled_dot_product(q, k, v)
print("Q\n", q)
print("K\n", k)
print("V\n", v)
print("Values\n", values)
print("Attention\n", attention)
```



```python
import torch
import torch.nn.functional as F

def dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

Transformer

### 数据的预处理

图像数据集
```python
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.StanfordCars(root=".", download=False, transform=DEFAULT_TRANSFORM)
torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=5)
```
文本数据集
```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb")["train"]
dataset = dataset.map(
	lambda sample: tokenizer(sample["text"], padding="max_length", truncation=True))
dataset.set_format(type="torch")
return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
```


训练时对整个数据集处理。
使用也需要预处理步骤。



### 其他

cnn

fcn


vgg

BatchNorm 在 线性层后激活层前
```python
self.net = nn.Sequential(
	nn.Conv2d(
		c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False
	),  # No bias needed as the Batch Norm handles it
	nn.BatchNorm2d(c_out),
	act_fn(),
	nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
	nn.BatchNorm2d(c_out),
)
```
resnet


核心之一是残差网络

```python
def forward(self, x):
	z = self.net(x)
	if self.downsample is not None:
		x = self.downsample(x)
	out = z + x
	out = self.act_fn(out)
	return out
```

DQN


```python
# DQN


```

评价指标

```
loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
acc = (preds.argmax(dim=-1) == labels).float().mean()
```


## 代码案例


- Hello world 	Pretrain - Hello world example 
- Image classification 	Finetune - ResNet-34 model to classify images of cars 
- Image segmentation 	Finetune - ResNet-50 model to segment images 
- Object detection 	Finetune - Faster R-CNN model to detect objects 
- Text classification 	Finetune - text classifier (BERT model) 
- Text summarization 	Finetune - text summarization (Hugging Face transformer model) 
- Audio generation 	Finetune - audio generator (transformer model) 
- LLM finetuning 	Finetune - LLM (Meta Llama 3.1 8B) 
- Image generation 	Pretrain - Image generator (diffusion model) 
- Recommendation system 	Train - recommendation system (factorization and embedding) 
- Time-series forecasting 	Train - Time-series forecasting with LSTM

### 张量操作

```python
import torch
import torch.nn as nn

x = torch.rand(5, 4) # 这里两个维度不一致，方便调试，对不上会报错。
# 通常第一维是 batch，最后一个维度是特征
# 即每行是一项数据，每列是一个特征
print(x)
print(x.shape) # [5, 4]

print(x.argmax(dim=-1)) # 每行得到一个下标，用于得到分类结果
print(x.mean(dim=-1)) # 合并特征
print(x.softmax(dim=-1)) # 如果仅仅要下标的话，这个可以不要。
# 如果是序列的话，有一个维度是序列长度。
# 如果是图片的话，特征有是多维。
# 特征合并用 concat 或者 +,通常最后一维是特征
print(torch.cat([x, 2*x], dim=-1)) # 特征维变长
print(x+2*x) # 特征维长度不变
# %%
# 如果图片的话，通道增加。
# 改变形状，有时候和广播配合。
print(x.unsqueeze(0).shape) # 添加 batch 或者 length 为 1 的时候
print(x.transpose(1, 0).shape) # 交换维度
print(x.view(-1, 2, 2).shape) # 进行 flatten 以及逆运算，注意行优先。

print(nn.Dropout(0.5)(x)) # 一部分特征变成 0. 用于传入 fc 前。

```

### CNN 图像分类
- CNN 常用卷积核 3， pool 后尺寸减半。
- 最后 flatten 后进入线性层
- 训练时对输入图像增强。
- pool 层缩小大小。
	- nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c_hidden[-1], self.hparams.num_classes)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # 也可以用 argmax
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')
    
```

### Resnet 微调

```python

```

### FastText 文本分类
FastText 文本分类
- 对文本向量取平均。速度比较快。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 定义 FastText 模型
class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        output = self.fc(pooled)
        return output


# 模拟数据
vocab_size = 1000
embedding_dim = 100
num_classes = 2
num_samples = 1000
max_length = 20

texts = [np.random.randint(0, vocab_size, max_length) for _ in range(num_samples)]
labels = np.random.randint(0, num_classes, num_samples)

# 创建数据集和数据加载器
dataset = TextDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = FastText(vocab_size, embedding_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for texts, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

```



### seq2seq
- rnn 需要对字符逐个编码，也需要逐个解码（直到结束标识）。实现为循环。
- rnn可以替换为lstm。
- 解码的损失函数是下一词的话可以并行。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 简单的词汇表
input_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'hello': 3, 'world': 4}
output_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'bonjour': 3, 'monde': 4}

input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)

# 示例数据
input_seqs = [['hello', 'world'], ['hello']]
target_seqs = [['bonjour', 'monde'], ['bonjour']]

# 将文本转换为索引序列
def text_to_indices(text, vocab):
    indices = [vocab['<SOS>']]
    for word in text:
        indices.append(vocab[word])
    indices.append(vocab['<EOS>'])
    return indices

input_indices = [text_to_indices(seq, input_vocab) for seq in input_seqs]
target_indices = [text_to_indices(seq, output_vocab) for seq in target_seqs]

# 填充序列
def pad_sequences(sequences, max_length):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            seq = seq + [input_vocab['<PAD>']] * (max_length - len(seq))
        padded_sequences.append(seq)
    return padded_sequences

max_input_length = max([len(seq) for seq in input_indices])
max_target_length = max([len(seq) for seq in target_indices])

input_padded = pad_sequences(input_indices, max_input_length)
target_padded = pad_sequences(target_indices, max_target_length)

# 转换为 PyTorch 张量
input_tensor = torch.tensor(input_padded, dtype=torch.long)
target_tensor = torch.tensor(target_padded, dtype=torch.long)

# 定义 LSTM 编码器
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

# 定义 LSTM 解码器
class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

# 训练参数
hidden_size = 256
encoder = EncoderLSTM(input_vocab_size, hidden_size)
decoder = DecoderLSTM(hidden_size, output_vocab_size)
criterion = nn.NLLLoss()
learning_rate = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

# 训练循环
n_iters = 100
for iter in range(n_iters):
    for i in range(len(input_tensor)):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_seq = input_tensor[i]
        target_seq = target_tensor[i]

        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_input_length, encoder.hidden_size)

        for ei in range(len(input_seq)):
            encoder_output, encoder_hidden = encoder(input_seq[ei].unsqueeze(0), encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[output_vocab['<SOS>']]], dtype=torch.long)
        decoder_hidden = encoder_hidden

        loss = 0
        for di in range(len(target_seq)):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_seq[di].unsqueeze(0))
            decoder_input = target_seq[di].unsqueeze(0)

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    if iter % 10 == 0:
        print(f'Iteration {iter}, Loss: {loss.item()}')

    
```

### RL 与 DQN


强化学习

这个示例展示了经典的 DQN 算法，包括：
- Q 网络的构建
- 经验回放缓冲区
- 目标网络更新
- ε- 贪婪策略

 
如果你想尝试其他环境，只需将 gym.make('CartPole-v1') 替换为其他环境名称即可。
 

```python
# !pip3 install "gym<=0.25.2"
# !pip3 install numpy==1.23.2
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# DQN 智能体
class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size=int(1e5), batch_size=64, seed=seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, 0.99)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# 训练函数
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

# 主程序
if __name__ == "__main__":
    # 导入必要的库
    import torch.nn.functional as F

    # 初始化环境和智能体
    env = gym.make('CartPole-v1')
    env.seed(0)
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

    # 训练智能体
    scores = dqn()

    # 绘制结果
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # 测试训练好的智能体
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    for i in range(3):
        state = env.reset()
        for j in range(200):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break
    env.close()
```

### Bert 微调


Hugging Face 可以用 pipeline 或者 AutoModel 来调用已有的模型。

以下是一个微调 Bert 用来文本分类的例子，使用了 Hugging Face 的训练框架（比如自带一些加速功能）。

类似的方法可以用于微调其他语言模型。

fine-tune
```python
# %%
# !pip3 install transformers datasets evaluate numpy scikit-learn 


#  加载 Yelp 评论数据集并进行 Tokenize 预处理
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载 Yelp 评论全量数据集（5 分类任务，标签 0-4 对应不同评分）
dataset = load_dataset("yelp_review_full")

# 加载预训练 Tokenizer（使用 BERT 基础模型的 cased 版本，保留大小写信息）
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# 定义 Tokenize 函数：处理文本padding（补全到最大长度）和truncation（截断超长文本）
def tokenize(examples):
    return tokenizer(
        examples["text"],  # 数据集的文本字段
        padding="max_length",  # 按模型最大输入长度补全（BERT 默认为 512）
        truncation=True  # 截断超过最大长度的文本
    )

# 批量处理全量数据集（batched=True 提升效率）
dataset = dataset.map(tokenize, batched=True)

# 抽取小样本子集（快速验证流程，正式训练可删除此步骤用全量数据）
# 训练集：随机打乱后取前 1000 条
small_train = dataset["train"].shuffle(seed=42).select(range(1000))
# 验证集：随机打乱后取前 1000 条
small_eval = dataset["test"].shuffle(seed=42).select(range(1000))

# 加载预训练模型（适配序列分类任务）
from transformers import AutoModelForSequenceClassification

# 加载 BERT 模型并指定分类任务的标签数量（Yelp 为 5 分类）
# 注意：模型会提示 "classifier" 层为新初始化，需通过微调训练
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased",
    num_labels=5  # 必须与任务标签数量一致
)

# 定义评估指标（计算分类任务的准确率）
import numpy as np
import evaluate

# 加载 accuracy 指标函数
metric = evaluate.load("accuracy")

# 定义指标计算函数：接收模型输出（logits）和真实标签，返回准确率
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # eval_pred 是模型预测输出与真实标签的元组
    predictions = np.argmax(logits, axis=-1)  # 将 logits 转为预测类别（取概率最大的类别）
    return metric.compute(predictions=predictions, references=labels)  # 计算并返回准确率

# 配置训练参数
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="yelp_review_classifier",  # 模型和训练日志的保存目录
    eval_strategy="epoch",  # 评估频率：每个 epoch 结束后评估一次
    # 以下为可选默认参数（可根据需求调整）：
    # learning_rate=2e-5,  # 学习率（预训练模型微调常用 2e-5 ~ 5e-5）
    # per_device_train_batch_size=8,  # 训练时每个设备的批次大小
    # per_device_eval_batch_size=8,  # 评估时每个设备的批次大小
    # num_train_epochs=3,  # 训练轮次
    # weight_decay=0.01,  # 权重衰减（防止过拟合）
)

# 初始化 Trainer 并启动训练
from transformers import Trainer

# 初始化 Trainer 实例（整合模型、参数、数据和评估函数）
trainer = Trainer(
    model=model,  # 待微调的模型
    args=training_args,  # 训练参数配置
    train_dataset=small_train,  # 训练数据集（正式训练可替换为 dataset["train"]）
    eval_dataset=small_eval,  # 评估数据集（正式训练可替换为 dataset["test"]）
    compute_metrics=compute_metrics,  # 评估指标计算函数
)

# 启动训练（自动执行训练、评估流程，按 eval_strategy 输出评估结果）
trainer.train()




# %%
import torch

# 准备输入文本
test_text = "The food was okay, but the service could be better."

# 对文本进行tokenize（返回PyTorch张量）
inputs = tokenizer(
    test_text,
    padding="max_length",
    truncation=True,
    return_tensors="pt"  # 返回PyTorch张量
)

# 模型推理（关闭梯度计算提高效率）
model.eval()  # 切换到评估模式
with torch.no_grad():
    outputs = model(**inputs)  # 传入tokenize后的输入

# 解析输出
logits = outputs.logits  # 获取原始输出（logits）
predictions = torch.argmax(logits, dim=-1).item()  # 转换为预测类别（0-4）
probabilities = torch.softmax(logits, dim=-1).tolist()[0]  # 转换为概率分布

print(f"预测类别: {predictions}")
print(f"类别概率: {[(i, round(p, 4)) for i, p in enumerate(probabilities)]}")

# %%
# 使用模型

from transformers import pipeline

# 加载训练好的模型和对应的tokenizer
classifier = pipeline(
    "text-classification",
    model="./yelp_review_classifier/checkpoint-375",  # 训练时指定的output_dir路径
    tokenizer=tokenizer  # 之前加载的BERT tokenizer
)

# 测试文本
test_texts = [
    "The food was okay, but the service could be better.",
    "This restaurant is amazing! The food was delicious and the service was excellent.",
    "Terrible experience. The food was cold and the staff was rude."
]

# 进行预测
results = classifier(test_texts)
print(results)

# %%

```
### ChatGPT

大语言模型

```python
from openai import OpenAI

# 配置客户端
client = OpenAI(
    api_key="YOUR_SECRET_KEY",
    base_url="http://127.0.0.1:11434/v1",
)

# 调用模型
response = client.chat.completions.create(
    model="qwen2.5:3b",  # 这里的模型名必须是你本地已有的
    messages=[
        {"role": "system", "content": "你是一个聪明的AI助手。"},
        {"role": "user", "content": "介绍一下你自己，以及你和 OpenAI 的关系。"},
    ],
    temperature=0.7,
)

# 打印结果
print(response.choices[0].message.content)
```

边生成边打印设置 `stream=True`, `
注意 chatgpt 的输入输出需要是聊天的格式。



RAG
- 向量数据库用来计算相似度进行检索，然后结合 chatgpt 来问答。

## 附录

参考资料
- https://lightning.ai/docs/pytorch/stable/tutorials.html
- [动手学深度学习](https://zh-v2.d2l.ai/)
- [Lambda calculus](https://crypto.stanford.edu/~blynn/lambda/)


原文创建于 2025年5月1日，随后略有调整。

### PyTorch 的用法

本文侧重案例分析，所以可能具体用法作为附录比较好，但是考虑本文的记录和阅读顺序，还是保留在开头了。

### 大语言模型

这个新的文档里再说，本文侧重传统的深度学习模型。


