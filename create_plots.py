import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve
from PIL import Image

start_dir = "results"
dir_name_list = [name for name in os.listdir(start_dir) if os.path.isdir(os.path.join(start_dir, name))]

if '.git' in dir_name_list:
  dir_name_list.remove(".git")

for dir_name in dir_name_list:
  save_name = os.path.join(start_dir, dir_name, f"{dir_name}.png")
  if os.path.isfile(save_name):
    print("Continued " + dir_name)
    continue

  train_dir = os.path.join(start_dir, dir_name, "train")
  test_dir = os.path.join(start_dir, dir_name, "val")

  epoch_numbers = list()
  train_f_scores = list()
  val_f_scores = list()

  test_files = [name for name in os.listdir(os.path.join(start_dir, dir_name)) if "test" in name and name.endswith(".csv")]
  available_folds = [int(name[-5]) for name in test_files]

  if len(test_files) == 0:
      continue

  train_csv_list = [f for f in os.listdir(os.path.join(train_dir, str(0))) if os.path.isfile(os.path.join(train_dir, str(0), f))]
  val_csv_list = [f for f in os.listdir(os.path.join(train_dir, str(0))) if os.path.isfile(os.path.join(test_dir, str(0), f))]

  for f in train_csv_list:
    epoch_num = int(f[-7:-4])
    epoch_numbers.append(epoch_num)

  for csv_train in train_csv_list:
    df = pd.DataFrame()
    for i in available_folds:
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

  for csv_val, e in zip(val_csv_list, epoch_numbers):
    df = pd.DataFrame()
    for i in available_folds:
      curr_file = os.path.join(test_dir, str(i), csv_val)
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

    val_f_scores.append(f_score)

  df = pd.DataFrame()
  for i in available_folds:
    curr_file = os.path.join(start_dir, dir_name, f"test_k{i}.csv")
    if not os.path.isfile(curr_file):
        continue
    df_curr = pd.read_csv(curr_file)
    df = pd.concat([df, df_curr])

  label = df['label'].to_numpy()
  model_score = df['model_score'].to_numpy()

  if len(label) == 0:
      test_score = 0.0
  else:
    precision, recall, thresholds = precision_recall_curve(label, model_score)

    f_score_list = list()
    for pre, rec in zip(precision, recall):
        lower = pre + rec
        if lower == 0.0:
            lower = 0.00000001
        
        f_score_list.append(2 * (pre * rec) / lower)

    test_score = max(f_score_list)

  teststr = f"Highest Val Score: {str(highest_f_score)[0:5]} @ epoch {highest_f_score_idx}\n"
  teststr = teststr + f"Test Score: {str(test_score)[0:5]}"
  
  zipped_data = zip(epoch_numbers, train_f_scores, val_f_scores)
  sorted_pairs = sorted(zipped_data)
  tuples = zip(*sorted_pairs)
  epoch_numbers, train_f_scores, val_f_scores = [ list(tuple) for tuple in  tuples]

  plt.plot(epoch_numbers, train_f_scores, color = "red", label = "Train")
  plt.plot(epoch_numbers, val_f_scores, color = "blue", label = "Val")
  plt.legend(loc="upper left")
  plt.title(f'Run Name: {dir_name}\n{teststr}')
  plt.ylabel('F-Score')
  plt.xlabel('Epoch Number')
  plt.savefig(save_name, bbox_inches='tight')
  print(dir_name)
  plt.close()