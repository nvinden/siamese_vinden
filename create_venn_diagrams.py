from fnmatch import fnmatch
import pandas as pd
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

#from itertools import count
import matplotlib.pyplot as plt
from metaphone import doublemetaphone
import pandas as pd
#from statistics import mean
import textdistance
import os
import numpy as np

from process import load_data, load_json_config

import random

import json
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

import os
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

results_list = [
        ["emb100_rat21_lstm_bi1_lay8", "emb100_rat21_gru_bi1_lay8", "21"],
        ["emb100_rat41_lstm_bi1_lay8", "emb100_rat41_gru_bi1_lay8", "41"]
    ]

def str2emb(string, string_pad = 30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_table = torch.full(size = (string_pad, ), fill_value = 32, device = device, dtype = torch.uint8)
    torch_table[0] = 30
    for j in range(1, len(string) + 1):
        cha = string[j - 1]
        torch_table[j] = ord(cha) - 97
    j += 1
    torch_table[j] = 31

    return torch_table

def emb2str(emb):
    word = ""
    for char in emb:
        char = char
        if char >= 30:
            continue
        word = word + chr(char + 97)
    return word

def create_ml_score(run_name):
    start_dir = "results/"

    data_list = list()

    try:
        dir_name = os.path.join(start_dir, run_name)
        json_name = os.path.join("data", run_name + ".json.json")

        folds = list()

        with open(json_name) as f:
            name_file = json.load(f)

            for fold_data in zip(name_file['positives'], name_file['random'], name_file['jeremy']):
                curr_fold_data = list()#pd.DataFrame(colums = ["name_a", "name_b", "label"])
                for i, data_now in enumerate(fold_data):
                    for curr_entry in data_now:
                        name_a = emb2str(curr_entry[0])
                        name_b = emb2str(curr_entry[1])
                        label = True if i == 0 else False

                        curr_fold_data.append({"name_a": name_a, "name_b": name_b, "label": label})
                df = pd.DataFrame(curr_fold_data)
                folds.append(df)

        def compare_dm1(s1, s2):
            return textdistance.levenshtein.normalized_similarity(doublemetaphone(s1)[0],doublemetaphone(s2)[0])

        def compare_dm2(s1, s2):
            return textdistance.levenshtein.normalized_similarity(doublemetaphone(s1)[1],doublemetaphone(s2)[1])

        for pairs in folds:
            pairs['levenshtein'] = [textdistance.levenshtein.normalized_similarity(x, y) for x, y in pairs[['name_a', 'name_b']].itertuples(index=False)]
            pairs['jaro'] = [textdistance.jaro.normalized_similarity(x, y) for x, y in pairs[['name_a', 'name_b']].itertuples(index=False)]
            pairs['jaro_winkler'] = [textdistance.jaro_winkler.normalized_similarity(x, y) for x, y in pairs[['name_a', 'name_b']].itertuples(index=False)]
            pairs['jaccard'] = [textdistance.jaccard.normalized_similarity(x, y) for x, y in pairs[['name_a', 'name_b']].itertuples(index=False)]
            pairs['sorensen_dice'] = [textdistance.sorensen_dice.normalized_similarity(x, y) for x, y in pairs[['name_a', 'name_b']].itertuples(index=False)]
            pairs['dm1'] = [compare_dm1(x, y) for x, y in pairs[['name_a', 'name_b']].itertuples(index=False)]
            pairs['dm2'] = [compare_dm2(x, y) for x, y in pairs[['name_a', 'name_b']].itertuples(index=False)]

            pairs['vowels_a'] = pairs['name_a'].apply(lambda x: sum(map(x.count, 'aeiou')))
            pairs['vowels_b'] = pairs['name_b'].apply(lambda x: sum(map(x.count, 'aeiou')))
            pairs['consonants_a'] = pairs['name_a'].str.len() - pairs['vowels_a']
            pairs['consonants_b'] = pairs['name_b'].str.len() - pairs['vowels_b']
            pairs['vowels'] = (pairs['vowels_a'] - pairs['vowels_b']).abs()
            pairs['vowels'] = 1 - (pairs['vowels'] / pairs['vowels'].max())
            pairs['consonants'] = (pairs['consonants_a'] - pairs['consonants_b']).abs()
            pairs['consonants'] = 1 - (pairs['consonants'] / pairs['consonants'].max())
            pairs['characters'] = (pairs['name_a'].str.len() - pairs['name_b'].str.len()).abs()
            pairs['characters'] = 1 - (pairs['characters'] / pairs['characters'].max())
            pairs = pairs.drop(columns=['vowels_a', 'vowels_b', 'consonants_a', 'consonants_b'])

        scores = list()
        y_prob = []

        for test_fold_index in range(len(folds)):
            val_fold_index = test_fold_index - 1 if test_fold_index - 1 >= 0 else len(folds) - 1

            X_train = pd.DataFrame()
            y_train = pd.Series(dtype=bool)
            for fold_index in range(len(folds)):
                if fold_index != test_fold_index and fold_index != val_fold_index:
                    X_train = pd.concat([X_train, folds[fold_index].drop(columns=['name_a', 'name_b', 'label'])])
                    y_train = pd.concat([y_train, folds[fold_index]['label']])
            names_test = folds[test_fold_index][['name_a', 'name_b']].to_numpy()
            X_test = folds[test_fold_index].drop(columns=['name_a', 'name_b', 'label'])
            y_test = folds[test_fold_index]['label']
            
            clf = RandomForestClassifier(random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            yt = np.array(y_test)
            yp = np.array(y_pred)

            tp = yt & yp
            tn = np.logical_not(yt) & np.logical_not(yp)
            fp = np.logical_not(yt) & yp
            fn = yt & np.logical_not(yp)

            assert len(tp) == len(tn)
            assert len(tp) == len(fp)
            assert len(tp) == len(fn)

            name_list = df[['name_a', 'name_b']].to_numpy()

            for i in range(len(tp)):
                names = names_test[i]
                scores.append({"name": f"{names[0]}_{names[1]}",
                    "tp": tp[i],
                    "tn": tn[i],
                    "fp": fp[i],
                    "fn": fn[i],
                    })

        return scores

    except Exception as e:
        print(run_name + " FAILED")
        print(e)
        return "error"

def get_name_lists(run_name):
    start_dir = "results/"

    data_list = list()

    try:
        dir_name = os.path.join(start_dir, run_name)
        json_name = os.path.join("data", run_name + ".json.json")

        folds = list()

        with open(json_name) as f:
            name_file = json.load(f)

            for fold_data in zip(name_file['positives'], name_file['random'], name_file['jeremy']):
                curr_fold_data = list()#pd.DataFrame(colums = ["name_a", "name_b", "label"])
                for i, data_now in enumerate(fold_data):
                    for curr_entry in data_now:
                        name_a = emb2str(curr_entry[0])
                        name_b = emb2str(curr_entry[1])
                        label = True if i == 0 else False

                        curr_fold_data.append({"name_a": name_a, "name_b": name_b, "label": label})
                folds.append(curr_fold_data)

        return folds

    except Exception as e:
        print(run_name + " FAILED")
        print(e)

def create_dl_scores(name_lists, lstm_name, gru_name, ratio):
    lstm_results_list = list()
    gru_results_list = list()

    for k in range(5):
        curr_list = name_lists[k]

        lstm_config_name = os.path.join("configs", f"{lstm_name}.json")
        gru_config_name = os.path.join("configs", f"{gru_name}.json")

        LSTM_DATASET_CONFIG, LSTM_TRAIN_CONFIG, LSTM_MODEL_KWARGS = load_json_config(lstm_config_name)
        GRU_DATASET_CONFIG, GRU_TRAIN_CONFIG, GRU_MODEL_KWARGS = load_json_config(gru_config_name)

        lstm_save_file = os.path.join("saves", f"{lstm_name}_k{k}_BEST")
        gru_save_file = os.path.join("saves", f"{gru_name}_k{k}_BEST")

        _, lstm_model, _, _, _, _, _ = load_data(lstm_save_file, LSTM_TRAIN_CONFIG, LSTM_MODEL_KWARGS)
        _, gru_model, _, _, _, _, _ = load_data(gru_save_file, GRU_TRAIN_CONFIG, GRU_MODEL_KWARGS)

        lstm_model = lstm_model.to(device)
        gru_model = gru_model.to(device)

        lstm_results_name = os.path.join("results", lstm_name, f"test_k{k}.csv")
        gru_results_name = os.path.join("results", gru_name, f"test_k{k}.csv")

        if not os.path.isfile(lstm_results_name):
            raise Exception(f"{lstm_results_name} NOT FOUND")

        if not os.path.isfile(gru_results_name):
            raise Exception(f"{gru_results_name} NOT FOUND")

        lstm_results = pd.read_csv(lstm_results_name).to_numpy()
        gru_results = pd.read_csv(gru_results_name).to_numpy()

        lstm_labels = lstm_results[:, 2].astype(dtype=bool)
        lstm_model_scores = lstm_results[:, 1]

        gru_labels = gru_results[:, 2].astype(dtype=bool)
        gru_model_scores = gru_results[:, 1]

        lstm_threshold = get_threshold(lstm_labels, lstm_model_scores)
        gru_threshold = get_threshold(gru_labels, gru_model_scores)

        for idx, data in enumerate(curr_list):
            name_a = data['name_a']
            name_b = data['name_b']
            label = data['label']

            name_a_emb = str2emb(name_a).unsqueeze(0)
            name_b_emb = str2emb(name_b).unsqueeze(0)

            mod_score, _ = lstm_model(name_a_emb, name_b_emb)
            mod_score = mod_score.item()

            pred_label = mod_score > lstm_threshold

            tp = pred_label and label
            fp = pred_label and not label
            tn = not pred_label and not label
            fn = not pred_label and label

            tp = bool(tp)
            fp = bool(fp)
            tn = bool(tn)
            fn = bool(fn)

            lstm_results_list.append({"name": f"{name_a}_{name_b}",
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    })
            
            mod_score, _ = gru_model(name_a_emb, name_b_emb)
            mod_score = mod_score.item()

            pred_label = mod_score > gru_threshold

            tp = pred_label and label
            fp = pred_label and not label
            tn = not pred_label and not label
            fn = not pred_label and label

            tp = bool(tp)
            fp = bool(fp)
            tn = bool(tn)
            fn = bool(fn)

            gru_results_list.append({"name": f"{name_a}_{name_b}",
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    })

    with open(ratio + ".json", 'w') as fout:
        json.dump([lstm_results_list, gru_results_list], fout)

def get_threshold(label, model_score):
    precision, recall, thresholds = precision_recall_curve(label, model_score)

    f_score_list = list()
    for pre, rec in zip(precision, recall):
        lower = pre + rec
        if lower == 0.0:
            lower = 0.00000001
        
        f_score_list.append(2 * (pre * rec) / lower)

    threshold_idx = np.argmax(f_score_list)

    return thresholds[threshold_idx]

for result_name in results_list:
    lstm_csv_file_names = os.path.join("results", result_name[0])
    gru_csv_file_names = os.path.join("results", result_name[1])

    name_lists = get_name_lists(result_name[0])

    ml_results = create_ml_score(result_name[0])

    with open(result_name[2] + ".json", "r") as f:
        lstm_results, gru_results = json.load(f)

    lstm_tp = {row['name'] for row in lstm_results if row['tp'] == True}
    lstm_fp = {row['name'] for row in lstm_results if row['fp'] == True}
    lstm_tn = {row['name'] for row in lstm_results if row['tn'] == True}
    lstm_fn = {row['name'] for row in lstm_results if row['fn'] == True}

    gru_tp = {row['name'] for row in gru_results if row['tp'] == True}
    gru_fp = {row['name'] for row in gru_results if row['fp'] == True}
    gru_tn = {row['name'] for row in gru_results if row['tn'] == True}
    gru_fn = {row['name'] for row in gru_results if row['fn'] == True}

    ml_tp = {row['name'] for row in ml_results if row['tp'] == True}
    ml_fp = {row['name'] for row in ml_results if row['fp'] == True}
    ml_tn = {row['name'] for row in ml_results if row['tn'] == True}
    ml_fn = {row['name'] for row in ml_results if row['fn'] == True}

    total_tp = lstm_tp.union(gru_tp).union(ml_tp)
    total_fp = lstm_fp.union(gru_fp).union(ml_fp)
    total_tn = lstm_tn.union(gru_tn).union(ml_tn)
    total_fn = lstm_fn.union(gru_fn).union(ml_fn)

    plt.figure(figsize=(8, 8))
    plt.title(f'True Positives for ratio {result_name[2]}')
    venn3(
        (lstm_tp, gru_tp, ml_tp),
        set_labels = ('LSTM', 'GRU', 'Random Forest'),
        subset_label_formatter=lambda x: f'{x:,}\n{x / len(total_tp) * 100:.1f}%'
    )
    plt.savefig(f"images/TP_{result_name[2]}.png")
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title(f'False Positives for ratio {result_name[2]}')
    venn3(
        (lstm_fp, gru_fp, ml_fp),
        set_labels = ('LSTM', 'GRU', 'Random Forest'),
        subset_label_formatter=lambda x: f'{x:,}\n{x / len(total_fp) * 100:.1f}%'
    )
    plt.savefig(f"images/FP_{result_name[2]}.png")
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title(f'True Negatives for ratio {result_name[2]}')
    venn3(
        (lstm_tn, gru_tn, ml_tn),
        set_labels = ('LSTM', 'GRU', 'Random Forest'),
        subset_label_formatter=lambda x: f'{x:,}\n{x / len(total_tn) * 100:.1f}%'
    )
    plt.savefig(f"images/TN_{result_name[2]}.png")
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title(f'False Negatives for ratio {result_name[2]}')
    venn3(
        (lstm_fn, gru_fn, ml_fn),
        set_labels = ('LSTM', 'GRU', 'Random Forest'),
        subset_label_formatter=lambda x: f'{x:,}\n{x / len(total_fn) * 100:.1f}%'
    )
    plt.savefig(f"images/FN_{result_name[2]}.png")
    plt.clf()
    #create_dl_scores(name_lists, result_name[0], result_name[1], result_name[2])

    print("hello")

