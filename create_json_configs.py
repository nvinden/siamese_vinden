import os
from process import load_json_config
import itertools
import json

session_number = 3

sample_json = os.path.join("configs", "run1_rat11_emb25.json")
std_DATASET, std_TRAIN, std_MODEL = load_json_config(sample_json)

if not os.path.isdir(os.path.join("configs", str(session_number))):
    os.mkdir(os.path.join("configs", str(session_number)))

hyperparam_list = [
    ['1', '2', '3', '4', '5'], #run type
    ['11', '21', '41'],        #ratio
    ['25', '50', '100']        #embedding
]

for hyp_combination in itertools.product(*hyperparam_list):
    run = hyp_combination[0]
    ratio = hyp_combination[1]
    emb = hyp_combination[2]

    DATASET = std_DATASET.copy()
    TRAIN = std_TRAIN.copy()
    MODEL = std_MODEL.copy()

    config_name = f"run{run}_rat{ratio}_emb{emb}.json"

    DATASET['partition_data_name'] = f"run{run}_rat{ratio}_emb{emb}"

    if emb == "25":
        MODEL["hidden_size"] = 25
    elif emb == "50":
        MODEL["hidden_size"] = 50
    elif emb == "100":
        MODEL["hidden_size"] = 100

    n_pos = 25000

    DATASET["test_random_negatives"] = int(n_pos / (5 * 2))
    DATASET["test_jeremy_negatives"] = int(n_pos / (5 * 2))

    if run == "1":
        DATASET["initial_random_negatives"] = int(1.0 * n_pos / 5)
        DATASET["initial_jeremy_negatives"] = int(0.0 * n_pos / 5)

        DATASET["test_random_negatives"] = int(1.0 * n_pos / 5)
        DATASET["test_jeremy_negatives"] = int(0.0 * n_pos / 5)

        TRAIN["active"] = False
        TRAIN["hard_neg_cap"] = int(0.0 * n_pos)
        DATASET['random_mutability'] = 0.0
        DATASET['jeremy_mutability'] = 0.0
    elif run == "2":
        DATASET["initial_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["initial_jeremy_negatives"] = int(0.5 * n_pos / 5)

        DATASET["test_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["test_jeremy_negatives"] = int(0.5 * n_pos / 5)

        TRAIN["active"] = False
        TRAIN["hard_neg_cap"] = int(0.0 * n_pos)
        DATASET['random_mutability'] = 0.0
        DATASET['jeremy_mutability'] = 0.0
    elif run == "3":
        DATASET["initial_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["initial_jeremy_negatives"] = int(0.0 * n_pos / 5)

        DATASET["test_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["test_jeremy_negatives"] = int(0.0 * n_pos / 5)

        TRAIN["active"] = True
        TRAIN["hard_neg_cap"] = int(0.6 * 0.5 * n_pos)
        DATASET['random_mutability'] = 0.0
        DATASET['jeremy_mutability'] = 0.0
    elif run == "4":
        DATASET["initial_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["initial_jeremy_negatives"] = int(0.5 * n_pos / 5)

        DATASET["test_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["test_jeremy_negatives"] = int(0.5 * n_pos / 5)

        TRAIN["active"] = True
        TRAIN["hard_neg_cap"] = int(0.6 * 0.5 * n_pos)
        DATASET['random_mutability'] = 0.5
        DATASET['jeremy_mutability'] = 0.5    
    elif run == "5":
        DATASET["initial_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["initial_jeremy_negatives"] = int(0.5 * n_pos / 5)

        DATASET["test_random_negatives"] = int(0.5 * n_pos / 5)
        DATASET["test_jeremy_negatives"] = int(0.5 * n_pos / 5)

        TRAIN["active"] = True
        TRAIN["hard_neg_cap"] = int(0.6 * 0.5 * n_pos)
        DATASET['random_mutability'] = 0.0
        DATASET['jeremy_mutability'] = 1.1

    if ratio == "11":
        DATASET["initial_random_negatives"] *= 1
        DATASET["initial_jeremy_negatives"] *= 1

        DATASET["test_random_negatives"] *= 1
        DATASET["test_jeremy_negatives"] *= 1

        TRAIN["hard_neg_cap"] *= 1
    elif ratio == "21":
        DATASET["initial_random_negatives"] *= 2
        DATASET["initial_jeremy_negatives"] *= 2

        DATASET["test_random_negatives"] *= 2
        DATASET["test_jeremy_negatives"] *= 2

        TRAIN["hard_neg_cap"] *= 2
    elif ratio == "41":
        DATASET["initial_random_negatives"] *= 4
        DATASET["initial_jeremy_negatives"] *= 4

        DATASET["test_random_negatives"] *= 4
        DATASET["test_jeremy_negatives"] *= 4
    
        TRAIN["hard_neg_cap"] *= 4

    out = {"DATASET_CONFIG": DATASET, "TRAIN_CONFIG": TRAIN, "MODEL_KWARGS": MODEL}
    config_path = os.path.join("configs", str(session_number), config_name)
    with open(config_path, 'w') as fp:
        json.dump(out, fp, indent=4)