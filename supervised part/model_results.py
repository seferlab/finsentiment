# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:41:40 2021

@author: Erva
"""
import pandas as pd
import numpy as np
from simpletransformers.model import TransformerModel, ClassificationModel
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import BertForSequenceClassification
class model_results:
    def __init__(self, model_type,j, model_path, t_file, v_file, epoch_num):
        self.model_type = model_type
        self.model_path =model_path
        self.t_file = t_file
        self.v_file = v_file
        self.epoch_num = epoch_num
        self.j = j
        
    def train(self):
        if ".csv" in self.t_file:
            train = pd.read_csv(self.t_file ,encoding="latin",header=None)#, names=['No','text','labels'])
            train.columns = ["text",'labels']
        if ".tsv" in self.t_file:
            train = pd.read_csv(self.t_file, sep='\t', names=['No','text','labels'])
            del train['No']
        train.labels +=1
        train = train[['text','labels']]
        model = ClassificationModel(self.model_type,self.model_path, num_labels=3,use_cuda=False, args={'learning_rate':1e-5, 'num_train_epochs': 10, 'reprocess_input_data': True, 'overwrite_output_dir': True,'save_model_every_epoch': False}) #)#('bert', '/content/drive/MyDrive/100mbout/1GB-bert'
        #model = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=3,use_cuda=False, args={'learning_rate':1e-5, 'num_train_epochs': 10, 'reprocess_input_data': True, 'overwrite_output_dir': True})
        model.train_model(train)
        return model
    def test(self):
        model1 = self.train()
        for k in range(len(self.v_file)):
                 
            val_file = self.v_file[k]
            if ".csv" in val_file:
                val = pd.read_csv(val_file ,encoding="latin",header=None)#, names=['No','text','labels'])
                val.columns = ["text",'labels']
            if ".tsv" in val_file:
                val = pd.read_csv(val_file, sep='\t', names=['No','text','labels'])
                del val['No']
            
            val = val[['text','labels']]
            val.labels +=1
            if k==self.j:
                df = val
                df['split'] = np.random.randn(df.shape[0], 1)
                msk = np.random.rand(len(df)) <= 0.7
                train = df[msk]
                val = df[~msk]               
                model = ClassificationModel(self.model_type,self.model_path, num_labels=3,use_cuda=False, args={'learning_rate':1e-5, 'num_train_epochs': 10, 'reprocess_input_data': True, 'overwrite_output_dir': True,'save_model_every_epoch': False}) #)#('bert', '/content/drive/MyDrive/100mbout/1GB-bert'
                #model = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=3,use_cuda=False, args={'learning_rate':1e-5, 'num_train_epochs': 10, 'reprocess_input_data': True, 'overwrite_output_dir': True})
                model.train_model(train)
            else:
                model = model1

            def f1_multiclass2(labels, preds):
                return f1_score(labels, preds, average='macro')
            def prec_score2(labels, preds):
                return precision_score(labels, preds, average='macro')
            def recall2(labels, preds):
                return recall_score(labels, preds, average='macro')
            result, model_outputs, wrong_predictions = model.eval_model(val,acc=accuracy_score, f1=f1_multiclass2, prec=prec_score2, recall = recall2)
            f = open("macro.txt", "a")
            f.write('\t TEST:'+val_file+'\n')
            f2 = open("./outputs/eval_results.txt",'r')
            f.write(f2.read())
            f2.close() 
            # for i in range(3):
            #     i=2
            #     if i ==0 :
            #         result, model_outputs, wrong_predictions = model.eval_model(val,acc=accuracy_score, f1=f1_multiclass, prec=prec_score, recall = recall)
            #         f = open("micro.txt", "a")
            #         f.write('\t TEST:'+val_file+'\n')
            #         f2 = open("./outputs/eval_results.txt",'r')
            #         f.write(f2.read())
            #         f2.close()
            #     elif i == 2:
            #         result, model_outputs, wrong_predictions = model.eval_model(val,acc=accuracy_score, f1=f1_multiclass2, prec=prec_score2, recall = recall2)
            #         f = open("macro.txt", "a")
            #         f.write('\t TEST:'+val_file+'\n')
            #         f2 = open("./outputs/eval_results.txt",'r')
            #         f.write(f2.read())
            #         f2.close()            
            #     else:
            #         result, model_outputs, wrong_predictions = model.eval_model(val,acc=accuracy_score, f1=f1_multiclass3, prec=prec_score3, recall = recall3)
            #         f = open("weighted.txt", "a")
            #         f.write('\t TEST:'+val_file+'\n')
            #         f2 = open("./outputs/eval_results.txt",'r')
            #         f.write(f2.read())
            #         f2.close()
