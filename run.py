import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForPreTraining
import torch.nn as nn
from seqeval.metrics import accuracy_score,f1_score
import time
from tqdm import tqdm
from matplotlib import pyplot as plt


# #忽略警告



# 读取数据
def readFile(name):
    data = []
    label = []
    dataSentence = []
    labelSentence = []
    with open(name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.strip():
                data.append(dataSentence)
                label.append(labelSentence)
                dataSentence = []
                labelSentence = []
            else:
                content = line.strip().split()
                dataSentence.append(content[0].lower())
                labelSentence.append(content[-1])
        # print(len(data))
        # print(data)
        # print(len(label))
        # print(label)
        # print(len(lines))
        return data, label

def label2index(label):
    label2index = {}
    for sentence in label:
        for i in sentence:
            if i not in label2index:
                label2index[i] = len(label2index)
    return label2index,list(label2index)


# 构建数据集
class Dataset(Dataset):
    def __init__(self, data, label, labelIndex, tokenizer, maxlength):
        self.data = data
        self.label = label
        self.labelIndex = labelIndex
        self.tokernizer = tokenizer
        self.maxlength = maxlength

    def __getitem__(self, item):
        thisdata = self.data[item]
        thislabel = self.label[item][:self.maxlength]
        thisdataIndex = self.tokernizer.encode(thisdata, add_special_tokens=True, max_length=self.maxlength + 2,
                                               padding="max_length", truncation=True, return_tensors="pt")
        thislabelIndex = [self.labelIndex['O']] + [self.labelIndex[i] for i in thislabel] + [self.labelIndex['O']] * (
                    maxlength + 1 - len(thislabel))
        thislabelIndex = torch.tensor(thislabelIndex)
        # print(thisdataIndex.shape)
        # print(thislabelIndex.shape)
        return thisdataIndex[-1], thislabelIndex,len(thislabel)

    def __len__(self):
        return self.data.__len__()


# 建模
class BertModel(nn.Module):
    def __init__(self, classnum, criterion):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained('bert-base-uncased').bert
        self.classifier = nn.Linear(768, classnum)
        self.criterion = criterion

    def forward(self, batchdata, batchlabel=None):
        bertOut=self.bert(batchdata)
        bertOut0,bertOut1=bertOut[0],bertOut[1]#字符级别bertOut[0].size()=torch.Size([batchsize, maxlength, 768]),篇章级别bertOut[1].size()=torch.Size([batchsize,768])
        pre=self.classifier(bertOut0)
        if batchlabel is not None:
            loss=self.criterion(pre.reshape(-1,pre.shape[-1]),batchlabel.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)
        # print(self.bert(batch_index))


if __name__ == '__main__':
    # 超参数
    batchsize = 64
    epoch = 100
    maxlength = 75
    lr = 0.01
    weight_decay = 0.00001

    # 读取数据
    trainData, trainLabel = readFile('train.txt')
    devData, devLabel = readFile('dev.txt')
    testData, testLabel = readFile('test.txt')

    # 构建词表
    labelIndex,indexLabel = label2index(trainLabel)
    # print(labelIndex)

    # 构建数据集,迭代器
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    trainDataset = Dataset(trainData, trainLabel, labelIndex, tokenizer, maxlength)
    trainDataloader = DataLoader(trainDataset, batch_size=batchsize, shuffle=False)
    devDataset = Dataset(devData, devLabel, labelIndex, tokenizer, maxlength)
    devDataloader = DataLoader(devDataset, batch_size=batchsize, shuffle=False)

    #建模
    criterion = nn.CrossEntropyLoss()
    model = BertModel(len(labelIndex), criterion).to(device)
    # print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # #绘图准备
    epochPlt = []
    trainLossPlt = []
    devAccPlt = []
    devF1Plt = []

    # 训练验证
    for e in range(epoch):
        #训练
        time.sleep(0.1)
        print(f'epoch:{e+1}')
        epochPlt.append(e+1)
        epochloss=0
        model.train()
        for batchdata, batchlabel,batchlen in tqdm(trainDataloader,total =len(trainDataloader),leave = False,desc="train"):
            batchdata=batchdata.to(device)
            batchlabel = batchlabel.to(device)
            loss=model.forward(batchdata, batchlabel)
            epochloss+=loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epochloss/=len(trainDataloader)
        trainLossPlt.append(float(epochloss))
        print(f'loss:{epochloss:.5f}')
            # print(batchdata.shape)
            # print(batchlabel.shape)

        #验证
        time.sleep(0.1)
        epochbatchlabel=[]
        epochpre=[]
        model.eval()
        for batchdata, batchlabel,batchlen in tqdm(devDataloader,total =len(devDataloader),leave = False,desc="dev"):
            batchdata=batchdata.to(device)
            batchlabel = batchlabel.to(device)
            pre=model.forward(batchdata)
            pre=pre.cpu().numpy().tolist()
            batchlabel = batchlabel.cpu().numpy().tolist()

            for b,p,l in zip(batchlabel,pre,batchlen):
                b=b[1:l+1]
                p=p[1:l+1]
                b=[indexLabel[i] for i in b]
                p=[indexLabel[i] for i in p]
                epochbatchlabel.append(b)
                epochpre.append(p)
            # print(pre)
        acc=accuracy_score(epochbatchlabel,epochpre)
        f1=f1_score(epochbatchlabel,epochpre)
        devAccPlt.append(acc)
        devF1Plt.append(f1)
        print(f'acc:{acc:.4f}')
        print(f'f1:{f1:.4f}')

        #绘图
        # print(epochPlt, trainLossPlt,devAccPlt,devF1Plt)
        plt.plot(epochPlt, trainLossPlt)
        plt.plot(epochPlt, devAccPlt)
        plt.plot(epochPlt, devF1Plt)
        plt.ylabel('loss/Accuracy/f1')
        plt.xlabel('epoch')
        plt.legend(['trainLoss', 'devAcc', 'devF1'], loc='best')
        plt.show()