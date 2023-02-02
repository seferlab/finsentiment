# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:10:43 2021

@author: Erva
"""

from model_results import model_results
import pandas as pd
from simpletransformers.model import TransformerModel
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import BertForSequenceClassification


if __name__ == "__main__":
    model_path =r'/home/erva_gc_user/roberta-base/output-roberta-base'
    for i in range(6):
        test_list = ['/home/erva_gc_user/fine_tune/sup/berttrainfiqapost2.tsv',
                     '/home/erva_gc_user/fine_tune/sup/berttrainfiqaheadline2.tsv',
                     '/home/erva_gc_user/fine_tune/sup/arrangedallagree.csv',
                     '/home/erva_gc_user/fine_tune/sup/arranged75agree.csv',
                     '/home/erva_gc_user/fine_tune/sup/arranged66agree.csv',
                     '/home/erva_gc_user/fine_tune/sup/arranged50agree.csv']

        t_file = test_list[i]
        if i ==3:
            break
        #test_list.pop(i)
        f = open("micro.txt", "a")
        f2 = open("macro.txt", "a")
        f3 = open("weighted.txt", "a")
        f.write("TRAIN: "+t_file+"\n")
        f2.write("TRAIN: "+t_file+ "\n")
        f3.write("TRAIN: "+t_file+"\n")
        f.close()
        f2.close()
        f3.close()
        
        model_results('roberta',i,model_path,t_file,test_list,20).test()
        #tokenizer, model_type, model_path, t_file, v_file, epoch_num)
