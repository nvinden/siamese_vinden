import torch
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import warnings
from PIL import Image
import json

from pyjarowinkler import distance as jw_dist

root_dir = "results"

dir_name_list = os.listdir(root_dir)

for dir_name in dir_name_list:
    try:
        image_save_file_name = os.path.join(root_dir, dir_name, "train_test_curve.png")
        if os.path.isfile(image_save_file_name):
            print(dir_name + " FOUND. skipping...")
            continue

        train_dir = os.path.join(root_dir, dir_name, "train")
        test_dir = os.path.join(root_dir, dir_name, "val")

        epoch_numbers = list()
        train_f_scores = list()
        test_f_scores = list()

        train_csv_list = [f for f in os.listdir(os.path.join(train_dir, str(0))) if os.path.isfile(os.path.join(train_dir, str(0), f))]
        test_csv_list = [f for f in os.listdir(os.path.join(train_dir, str(0))) if os.path.isfile(os.path.join(test_dir, str(0), f))]

        for f in train_csv_list:
            epoch_num = int(f[-7:-4])
            epoch_numbers.append(epoch_num)

        for csv_train in train_csv_list:
            df = pd.DataFrame()
            for i in range(5):
                curr_file = os.path.join(train_dir, str(i), csv_train)
                if not os.path.isfile(curr_file):
                    continue
                df_curr = pd.read_csv(curr_file)
                df = pd.concat([df, df_curr])

            label = df['label'].to_numpy()
            model_score = df['model_score'].to_numpy()

            precision, recall, thresholds = precision_recall_curve(label, model_score)

            f_score_list = list()
            for pre, rec in zip(precision, recall):
                lower = pre + rec
                if lower == 0.0:
                    lower = 0.00000001
                
                f_score_list.append(2 * (pre * rec) / lower)

                f_score = max(f_score_list)

            train_f_scores.append(f_score)
        
        highest_f_score = 0
        highest_f_score_idx = 0

        for csv_test, e in zip(test_csv_list, epoch_numbers):
            df = pd.DataFrame()
            for i in range(5):
                curr_file = os.path.join(test_dir, str(i), csv_test)
                df_curr = pd.read_csv(curr_file)
                df = pd.concat([df, df_curr])

            label = df['label'].to_numpy().astype(np.int8)
            model_score = df['model_score'].to_numpy()

            precision, recall, thresholds = precision_recall_curve(label, model_score)

            f_score_list = list()
            for pre, rec in zip(precision, recall):
                lower = pre + rec
                if lower == 0.0:
                    lower = 0.00000001
            
            f_score_list.append(2 * (pre * rec) / lower)

            f_score = max(f_score_list)

            if f_score > highest_f_score:
                highest_f_score = f_score
                highest_f_score_idx = e

            test_f_scores.append(f_score)

        teststr = f"Highest Val Score: {str(highest_f_score)[0:5]} @ epoch {highest_f_score_idx}"
        

        print(epoch_numbers)
        plt.plot(epoch_numbers, train_f_scores, color = "red", label = "Train")
        plt.plot(epoch_numbers, test_f_scores, color = "blue", label = "Val")
        plt.legend(loc="upper left")
        plt.title(f'Run Name: {dir_name[:-1]}\n{teststr}')
        plt.ylabel('F-Score')
        plt.xlabel('Epoch Number')
        plt.savefig(os.path.join(root_dir, dir_name, "train_test_curve.png"))
        plt.close()
        print(dir_name)
    except OSError as e:
        print(dir_name + " NOT FOUND")