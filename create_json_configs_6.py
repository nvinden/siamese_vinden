import os
from process import load_json_config
import itertools
import json

session_number = 6

sample_json = os.path.join("configs", "test.json")
std_DATASET, std_TRAIN, std_MODEL = load_json_config(sample_json)

if not os.path.isdir(os.path.join("configs", str(session_number))):
    os.mkdir(os.path.join("configs", str(session_number)))

hyperparam_list = [
    [5, 10, 15, 20, 25],                 #emb
    ['edit', 'hard', 'random', 'nick'],  #run type
    [4, 8, 12, 16]                       #layers
]

for hyp_combination in itertools.product(*hyperparam_list):
    emb = hyp_combination[0]
    run = hyp_combination[1]
    lay = hyp_combination[2]

    DATASET = std_DATASET.copy()
    TRAIN = std_TRAIN.copy()
    MODEL = std_MODEL.copy()

    TRAIN["hnm_period"] = 20

    config_name = f"{run}_emb{emb}_lay{lay}.json"

    DATASET['partition_data_name'] = config_name

    MODEL["hidden_size"] = emb

    if run == 'edit':
        DATASET["initial_random_negatives"] = 5000
        DATASET["initial_jeremy_negatives"] = 5000
        DATASET["test_random_negatives"] = 5000
        DATASET["test_jeremy_negatives"] = 5000

        DATASET['active'] = False
        DATASET['predef_weights'] = False
    if run == 'random':
        DATASET["initial_random_negatives"] = 5000
        DATASET["initial_jeremy_negatives"] = 0
        DATASET["test_random_negatives"] = 5000
        DATASET["test_jeremy_negatives"] = 0

        DATASET['active'] = False
        DATASET['predef_weights'] = False
    if run == 'hard':
        DATASET["initial_random_negatives"] = 5000
        DATASET["initial_jeremy_negatives"] = 5000
        DATASET["test_random_negatives"] = 5000
        DATASET["test_jeremy_negatives"] = 5000

        DATASET['active'] = True
        DATASET['predef_weights'] = False
    if run == 'nick':
        DATASET["initial_random_negatives"] = 5000
        DATASET["initial_jeremy_negatives"] = 5000
        DATASET["test_random_negatives"] = 5000
        DATASET["test_jeremy_negatives"] = 5000

        DATASET['active'] = False
        DATASET['predef_weights'] = True

    MODEL['num_layers'] = lay

    out = {"DATASET_CONFIG": DATASET, "TRAIN_CONFIG": TRAIN, "MODEL_KWARGS": MODEL}
    config_path = os.path.join("configs", str(session_number), config_name)
    with open(config_path, 'w') as fp:
        json.dump(out, fp, indent=4)