# -*- Encoding:UTF-8 -*-

import numpy as np
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math

import torch
import torch.autograd
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
import time

use_gpu = torch.cuda.is_available()

def useGpu(input):
    if use_gpu:
        input = input.cuda()
    return input 

allTime = time.localtime(time.time())
outfileTime = str(allTime[0]*10000 + allTime[1]*100 +allTime[2])

def main():
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-dataName', action='store', dest='dataName', default='ml_100k.csv')
    parser.add_argument('-splitSign', action='store', dest='splitSign', default=',')
    parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.0001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)
    parser.add_argument('-mu', action='store', dest='mu', default=1e-6)

    args = parser.parse_args()
    classifier = Model(args)
    optimizer = addOptimizer(classifier)
    iteration = 25
    for i in range(iteration):
        with open(outfileTime + '.txt', 'a') as f:
            f.write('iteration {}\n'.format(i))
        print("+"*20+"iteration {}".format(i)+"+"*20)
        run(classifier, optimizer)
    


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.dataName = args.dataName
        self.splitSign = args.splitSign
        self.dataSet = DataSet(self.dataName, self.splitSign)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate
        self.mu= args.mu

        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.negNum = args.negNum
        self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        self.add_embedding_matrix()

        # self.add_placeholders()

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        # self.add_model()

        # self.add_loss()

        self.lr = args.lr
        # self.add_train_step()

        self.checkPoint = args.checkPoint
        # self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop
        self.init_model()
        self.training = True

    def init_model(self):
        self.User_Layer = torch.nn.Sequential(
            #2
            # nn.Linear(self.shape[1], self.userLayer[0]),
            # nn.Dropout(0.15),
            # nn.ReLU(True),
            # nn.Linear(self.userLayer[0], self.userLayer[1]),
            # nn.Dropout(0.1),
            # nn.ReLU(True),
            # nn.Linear(self.userLayer[1], self.userLayer[2]),
            # nn.ReLU(True)

            #1
            # nn.Linear(self.shape[1], self.userLayer[0]),
            # nn.ReLU(True),
            # nn.Linear(self.userLayer[0], self.userLayer[1]),
            # nn.ReLU(True)

            #3
            nn.Linear(self.shape[1], self.userLayer[0]),
            nn.Tanh(),
            nn.Linear(self.userLayer[0], self.userLayer[1]),
            nn.Tanh()
        )
        
        self.Item_Layer = torch.nn.Sequential(
            #2
            # nn.Linear(self.shape[0], self.itemLayer[0]),
            # nn.Dropout(0.15),
            # nn.ReLU(True),
            # nn.Linear(self.itemLayer[0], self.itemLayer[1]),
            # nn.Dropout(0.1),
            # nn.ReLU(True),
            # nn.Linear(self.itemLayer[1], self.itemLayer[2]),
            # nn.ReLU(True)
            #1
            # nn.Linear(self.shape[0], self.itemLayer[0]),
            # nn.Tanh(),
            # nn.Linear(self.itemLayer[0], self.itemLayer[1]),
            # nn.ReLU(True)

            #3
            nn.Linear(self.shape[0], self.itemLayer[0]),
            nn.Tanh(),
            nn.Linear(self.itemLayer[0], self.itemLayer[1]),
            nn.Tanh()

        )

        for param in self.parameters():
            nn.init.normal(param,0,0.01)
        

    def add_embedding_matrix(self):
        # self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        # self.item_user_embedding = tf.transpose(self.user_item_embedding)
        self.user_item_embedding = torch.from_numpy(self.dataSet.getEmbedding())
        self.item_user_embedding = torch.t(self.user_item_embedding)

    def forward(self, userI, itemJ):
        p = self.User_Layer(userI)
        q = self.Item_Layer(itemJ)
        

        y_pre = F.cosine_similarity(p, q)
        # tensor_exp = torch.FloatTensor(y_pre.size()).fill_(math.exp(1)) 
        # tensor_exp = useGpu(tensor_exp)
        # y_pre.data = y_pre.data.exp().div(tensor_exp)
        
        tensor_mu = torch.FloatTensor(y_pre.size()).fill_(self.mu) 
        tensor_mu = useGpu(tensor_mu)
        y_pre.data = y_pre.data.max(tensor_mu)

        return y_pre


def defCrossEntropy(y_pre, y_true):
    loss = F.binary_cross_entropy(y_pre, y_true)
    return loss

def addOptimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    return optimizer
    
def run(model, optimizer):
    best_hr = -1
    best_NDCG = -1
    best_epoch = -1
    print("Start Training!")
    for epoch in range(model.maxEpochs):
        print("="*20+"Epoch {}".format(epoch)+"="*20)
        model.training = True
        run_epoch(model, optimizer)
        print('='*50)
        print("Start Evaluation!")
        model.training = True
        hr, NDCG = evaluate(model, model.topK)
        print("Epoch ".format(epoch) + "HR: {}, NDCG: {}".format(hr, NDCG))
        if hr > best_hr or NDCG > best_NDCG:
            best_hr = hr
            best_NDCG = NDCG
            best_epoch = epoch
            if not os.path.exists('../model/'):
                os.mkdir('../model/')

            torch.save(model.state_dict(), '../model/' + model.dataName + str(epoch)+'.pth')    
        
        if epoch - best_epoch > model.earlyStop:
            print("Normal Early stop!")
            break
        print("="*20+"Epoch ".format(epoch)+"End"+"="*20)
    with open(outfileTime + '.txt', 'a') as f:
        f.write("Best hr: {}, NDCG: {}, At Epoch {}\n".format(best_hr, best_NDCG, best_epoch))
        
    print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
    print("Training complete!")


def run_epoch(model, optimizer, verbose=10):
    train_u, train_i, train_r = model.dataSet.getInstances(model.train, model.negNum)
    train_len = len(train_u)
    shuffled_idx = np.random.permutation(np.arange(train_len))
    train_u = train_u[shuffled_idx]
    train_i = train_i[shuffled_idx]
    train_r = train_r[shuffled_idx]

    num_batches = len(train_u) // model.batchSize + 1

    losses = []
    optimizer.zero_grad()
    lastTime = time.time()
    for i in range(num_batches):
        optimizer.zero_grad()
        min_idx = i * model.batchSize
        max_idx = np.min([train_len, (i+1)*model.batchSize])
        train_u_batch = train_u[min_idx: max_idx]
        train_i_batch = train_i[min_idx: max_idx]
        train_r_batch = train_r[min_idx: max_idx]

        user_scores = torch.index_select(model.user_item_embedding, 0, torch.from_numpy(train_u_batch).type(torch.LongTensor))
        item_scores = torch.t(torch.index_select(model.user_item_embedding, 1, torch.from_numpy(train_i_batch).type(torch.LongTensor)))
        y_true = torch.from_numpy(train_r_batch).type(torch.FloatTensor)
        y_true = useGpu(y_true)
        user_scores = useGpu(user_scores)
        item_scores = useGpu(item_scores)

        y_pre = model(Variable(user_scores), Variable(item_scores))
        
        regRate = Variable(torch.div(y_true, model.maxRate))
        
        loss = defCrossEntropy(y_pre, regRate)

        losses.append(loss.data.sum())
        loss.backward()
        optimizer.step()

        if verbose and i % verbose == 0:
            newTime = time.time()
            sys.stdout.write('{} / {} : loss = {}, cost time = {}\r'.format(
                i, num_batches, np.mean(losses[-verbose:]), newTime-lastTime
            ))
            lastTime = newTime
            sys.stdout.flush()

    lo = np.mean(losses)
    print("\nMean loss in this epoch is: {}".format(lo))
    return loss

def evaluate(model, topK):
    def getHitRatio(ranklist, targetItem):
        for item in ranklist:
            if item == targetItem:
                return 1
        return 0
    def getNDCG(ranklist, targetItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == targetItem:
                return math.log(2) / math.log(i+2)
        return 0


    hr =[]
    NDCG = []
    testUser = model.testNeg[0]
    testItem = model.testNeg[1]
    for i in range(len(testUser)):
        target = testItem[i][0]
        user_scores = model.user_item_embedding.index_select(0, torch.from_numpy(testUser[i]).type(torch.LongTensor))
        item_scores = model.item_user_embedding.index_select(0, torch.from_numpy(testItem[i]).type(torch.LongTensor))
        user_scores = useGpu(user_scores)
        item_scores = useGpu(item_scores)

        predi_scores = model(Variable(user_scores), Variable(item_scores))
        predict = []
        for data in predi_scores.data:
            predict.append(data)

        item_score_dict = {}

        for j in range(len(testItem[i])):
            item = testItem[i][j]
            item_score_dict[item] = predict[j]

        ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

        tmp_hr = getHitRatio(ranklist, target)
        tmp_NDCG = getNDCG(ranklist, target)
        hr.append(tmp_hr)
        NDCG.append(tmp_NDCG)
    with open(outfileTime + '.txt', 'a') as f:
        f.write('HR {}'.format(np.mean(hr)) + ' NDCG {}\n'.format(np.mean(NDCG)))
    return np.mean(hr), np.mean(NDCG)



if __name__ == '__main__':
    main()

#     def add_placeholders(self):
#         self.user = tf.placeholder(tf.int32)
#         self.item = tf.placeholder(tf.int32)
#         self.rate = tf.placeholder(tf.float32)
#         self.drop = tf.placeholder(tf.float32)

    
#     def add_model(self):
#         user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
#         item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

#         def init_variable(shape, name):
#             return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

#         with tf.name_scope("User_Layer"):
#             user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
#             user_out = tf.matmul(user_input, user_W1)
#             for i in range(0, len(self.userLayer)-1):
#                 W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
#                 b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
#                 user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

#         with tf.name_scope("Item_Layer"):
#             item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
#             item_out = tf.matmul(item_input, item_W1)
#             for i in range(0, len(self.itemLayer)-1):
#                 W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
#                 b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
#                 item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

#         norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
#         norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
#         self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)
#         self.y_ = tf.maximum(1e-6, self.y_)


#     def add_loss(self):
#         regRate = self.rate / self.maxRate
#         losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
#         loss = -tf.reduce_sum(losses)
#         # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#         # self.loss = loss + self.reg * regLoss
#         self.loss = loss

#     def add_train_step(self):
#         '''
#         global_step = tf.Variable(0, name='global_step', trainable=False)
#         self.lr = tf.train.exponential_decay(self.lr, global_step,
#                                              self.decay_steps, self.decay_rate, staircase=True)
#         '''
#         optimizer = tf.train.AdamOptimizer(self.lr)
#         self.train_step = optimizer.minimize(self.loss)

#     def init_sess(self):
#         self.config = tf.ConfigProto()
#         self.config.gpu_options.allow_growth = True
#         self.config.allow_soft_placement = True
#         self.sess = tf.Session(config=self.config)
#         self.sess.run(tf.global_variables_initializer())

#         self.saver = tf.train.Saver()
#         if os.path.exists(self.checkPoint):
#             [os.remove(f) for f in os.listdir(self.checkPoint)]
#         else:
#             os.mkdir(self.checkPoint)

#     def run(self):
#         best_hr = -1
#         best_NDCG = -1
#         best_epoch = -1
#         print("Start Training!")
#         for epoch in range(self.maxEpochs):
#             print("="*20+"Epoch ", epoch, "="*20)
#             self.run_epoch(self.sess)
#             print('='*50)
#             print("Start Evaluation!")
#             hr, NDCG = self.evaluate(self.sess, self.topK)
#             print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
#             if hr > best_hr or NDCG > best_NDCG:
#                 best_hr = hr
#                 best_NDCG = NDCG
#                 best_epoch = epoch
#                 self.saver.save(self.sess, self.checkPoint)
#             if epoch - best_epoch > self.earlyStop:
#                 print("Normal Early stop!")
#                 break
#             print("="*20+"Epoch ", epoch, "End"+"="*20)
#         print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
#         print("Training complete!")

#     def run_epoch(self, sess, verbose=10):
#         train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
#         train_len = len(train_u)
#         shuffled_idx = np.random.permutation(np.arange(train_len))
#         train_u = train_u[shuffled_idx]
#         train_i = train_i[shuffled_idx]
#         train_r = train_r[shuffled_idx]

#         num_batches = len(train_u) // self.batchSize + 1

#         losses = []
#         for i in range(num_batches):
#             min_idx = i * self.batchSize
#             max_idx = np.min([train_len, (i+1)*self.batchSize])
#             train_u_batch = train_u[min_idx: max_idx]
#             train_i_batch = train_i[min_idx: max_idx]
#             train_r_batch = train_r[min_idx: max_idx]

#             feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
#             _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
#             losses.append(tmp_loss)
#             if verbose and i % verbose == 0:
#                 sys.stdout.write('\r{} / {} : loss = {}'.format(
#                     i, num_batches, np.mean(losses[-verbose:])
#                 ))
#                 sys.stdout.flush()
#         loss = np.mean(losses)
#         print("\nMean loss in this epoch is: {}".format(loss))
#         return loss

#     def create_feed_dict(self, u, i, r=None, drop=None):
#         return {self.user: u,
#                 self.item: i,
#                 self.rate: r,
#                 self.drop: drop}

#     def evaluate(self, sess, topK):
#         def getHitRatio(ranklist, targetItem):
#             for item in ranklist:
#                 if item == targetItem:
#                     return 1
#             return 0
#         def getNDCG(ranklist, targetItem):
#             for i in range(len(ranklist)):
#                 item = ranklist[i]
#                 if item == targetItem:
#                     return math.log(2) / math.log(i+2)
#             return 0


#         hr =[]
#         NDCG = []
#         testUser = self.testNeg[0]
#         testItem = self.testNeg[1]
#         for i in range(len(testUser)):
#             target = testItem[i][0]
#             feed_dict = self.create_feed_dict(testUser[i], testItem[i])
#             predict = sess.run(self.y_, feed_dict=feed_dict)

#             item_score_dict = {}

#             for j in range(len(testItem[i])):
#                 item = testItem[i][j]
#                 item_score_dict[item] = predict[j]

#             ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

#             tmp_hr = getHitRatio(ranklist, target)
#             tmp_NDCG = getNDCG(ranklist, target)
#             hr.append(tmp_hr)
#             NDCG.append(tmp_NDCG)
#         return np.mean(hr), np.mean(NDCG)

# if __name__ == '__main__':
#     main()

