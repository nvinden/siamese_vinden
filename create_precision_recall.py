import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, f1_score

import pandas as pd

from os import listdir
from os.path import isfile, join

from process import load_json_config

if __name__ == "__main__":
    columns = ["run_name", "avg_precision_model", "avg_precision_JW"]
    save_to = pd.DataFrame(columns = columns)
    save_to.astype({'avg_precision_model': 'float64'}).dtypes
    save_to.astype({'avg_precision_JW': 'float64'}).dtypes

    csv_files = [f for f in listdir("results") if isfile(join("results", f))]
    for file in csv_files:
        if "Zone" in file:
            continue

        json_file = join("configs", file.replace(".csv", ".json"))
        DATASET_CONFIG, TRAIN_CONFIG, MODEL_KWARGS = load_json_config(json_file)


        results = pd.read_csv(join("results", file))

        training_initial_random_negatives = int(DATASET_CONFIG['initial_random_negatives'] / DATASET_CONFIG['k'] * (DATASET_CONFIG['k'] - 1))
        training_initial_jeremy_negatives = int(DATASET_CONFIG['initial_jeremy_negatives'] / DATASET_CONFIG['k'] * (DATASET_CONFIG['k'] - 1))

        testing_initial_jeremy_negatives = int(DATASET_CONFIG['test_jeremy_negatives'] / DATASET_CONFIG['k'])
        testing_initial_random_negatives = int(DATASET_CONFIG['test_random_negatives'] / DATASET_CONFIG['k'])

        kth = DATASET_CONFIG['kth_example']

        if "bidirectional" in MODEL_KWARGS:
            bidirectional = MODEL_KWARGS['bidirectional']
        else:
            bidirectional = False

        lr = TRAIN_CONFIG['lr']
        n_epochs = TRAIN_CONFIG["n_epochs"]
        A_name = TRAIN_CONFIG["A_name"]

        model_score = results['model_score'].to_numpy()
        JW_score = results["JW_scores"].to_numpy()
        label = results['label'].to_numpy()

        jw_precision, jw_recall, jw_threshold  = precision_recall_curve(label, JW_score)
        jw_average = average_precision_score(label, JW_score)
        #jw_F = f1_score(label, JW_score)

        mod_precision, mod_recall, mod_threshold  = precision_recall_curve(label, model_score)
        mod_average = average_precision_score(label, model_score)
        #mod_F = f1_score(label.astype(int), model_score, average='macro')

        save_dict = {
            "run_name": file, 
            "avg_precision_model": mod_average, 
            "avg_precision_JW": jw_average,
            "training_initial_random_negatives": training_initial_random_negatives,
            "training_initial_jeremy_negatives": training_initial_jeremy_negatives,
            "testing_initial_jeremy_negatives": testing_initial_jeremy_negatives,
            "testing_initial_random_negatives": testing_initial_random_negatives,
            "kth": kth,
            "bidirectional": bidirectional,
            "lr": lr,
            "n_epochs": n_epochs,
            "A_name": A_name
        }

        save_to = save_to.append(save_dict, ignore_index = True)

    save_to.to_csv("precision_recall.csv")