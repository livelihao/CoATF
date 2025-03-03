#coding:utf-8
import time
import config
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IDMapper():
    def __init__(self):
        self.rawID2ID = {}
        self.rawIDs = []
    
    def getId(self, rawID, addNew=True):
        ID = self.rawID2ID.get(rawID, -1)

        if ID == -1 and addNew:
            ID = len(self.rawIDs) + 1
            self.rawID2ID[rawID] = ID
            self.rawIDs.append(rawID)
        return ID        

    def returnrawID2ID(self):
        return self.rawID2ID

    def getRawID(self, ID):
        return self.rawIDs[ID]
    
    def getNumIDs(self):
        return len(self.rawIDs)

class DataHandler():
    def __init__(self):
        self.userIDMapper = IDMapper()
        self.itemIDMapper = IDMapper()
        self.contIDMapper = IDMapper()
        self.trainData_x = []
        self.trainData_y = []

        self.testData_x = []
        self.testData_y = []
        self.mu = 0

        self.userRatedItems = defaultdict(list)


    def loadData(self, file_path, labelColIndices=None):
        """
        file_path: support multi file path seperated by ',' like: file1,file2,file3,...
        """
        data = []
        file_paths = file_path.split(',')
        for single_path in file_paths:
            with open(single_path) as lines:
                for line in lines:
                    if config.dataset == 'Ciao':
                        (userID, movieID, genreID, reviewID, movieRating, date) = line.strip().split('\t')  # Ciao
                    elif config.dataset == 'epinions':
                        (userID, movieID, genreID, movieRating, reviewID, date) = line.strip().split(',') # Epinions
                    elif(config.dataset == 'mov100k'):
                        (userID, movieID, movieRating, date) = line.strip().split('\t') # Mov100k
                    elif(config.dataset in ['Synthetic-r-10', 'Synthetic-r-200', 'Synthetic-r-4-NL']):
                        (userID, movieID, date, movieRating) = line.strip().split('\t') # 
                    elif config.dataset == 'Synthetic':
                        (userID, movieID, date, movieRating) = line.strip().split(' ') # 
                    elif config.dataset == 'gowalla':
                        (userID, movieID, date, latitude, longitude, movieRating) = line.strip().split('	')  # gollo
                    else:
                        pass
                    if config.dataset in ['Synthetic-r-10', 'Synthetic', 'Synthetic-r-200', 'Synthetic-r-4-NL']:
                        context = date
                    else:
                        # context = time.gmtime(float(date)).tm_wday # 周
                        context = time.gmtime(float(date)).tm_mday # 月
                    data.append((userID, movieID, float(movieRating), context))
        return data

    def align_Data(self, data, Training=True):
        '''
        :param data: 数据
        :param Training: 判断是不是训练集
        :return:
        '''
        n = 0
        num_rating = 0.0
        for sample in data:
            (rawUserId, rawItemId, rating, rawContId) = sample
            userId = self.userIDMapper.getId(rawUserId, Training) - 1
            itemId = self.itemIDMapper.getId(rawItemId, Training) - 1
            contId = self.contIDMapper.getId(rawContId, Training) - 1
            num_rating += rating
            n += 1
            if Training:
                self.trainData_x.append((userId, itemId,  contId))
                self.trainData_y.append(rating)
                self.userRatedItems[userId].append(itemId)
            else:
                if userId == -2 or itemId == -2 or contId == -2:
                    continue
                self.testData_x.append((userId, itemId, contId))
                self.testData_y.append(rating)
                self.userRatedItems[userId].append(itemId)
        if Training:
            self.mu = num_rating / n

def load_data(train_ratio):
    ############ read_data
    dataHandler = DataHandler()
    all_data = dataHandler.loadData(config.file_path)
    random.shuffle(all_data) # 打乱

    n_total = len(all_data)
    print("可观测到的条目", n_total)

    offsef = int(n_total * train_ratio)
    train_data = all_data[:offsef]
    test_data = all_data[offsef:]
    
    return train_data, test_data, n_total

def RegLoss(model):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.norm(param, p=2)
    return reg_loss

# load_data()