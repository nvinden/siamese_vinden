import os
from process import load_json_config
import itertools
import json

session_number = 5

sample_json = os.path.join("configs", "run4_rat11_emb25.json")
std_DATASET, std_TRAIN, std_MODEL = load_json_config(sample_json)

if not os.path.isdir(os.path.join("configs", str(session_number))):
    os.mkdir(os.path.join("configs", str(session_number)))

hyperparam_list = [
    [25, 50, 100],              #emb
    ['11', '21', '41'],         #ratio
    ["gru", "lstm"],            #model_type
    [1, 0],                     #bidirectionality
    [1, 2, 4, 8]                #layers
]

lstm_config = {
    "input_size": 26,
    "hidden_size": 25,
    "num_layers": 1,
    "bias": True,
    "batch_first": True,
    "dropout": 0,
    "bidirectional": True
}
gru_config = {
    "input_size": 26,
    "hidden_size": 25,
    "num_layers": 1,
    "bias": True,
    "batch_first": True,
    "dropout": 0,
    "bidirectional": True
}

for hyp_combination in itertools.product(*hyperparam_list):
    emb = hyp_combination[0]
    ratio = hyp_combination[1]
    model = hyp_combination[2]
    bidir = hyp_combination[3]
    layer = hyp_combination[4]

    if bidir == 0 and layer != 1:
        continue

    DATASET = std_DATASET.copy()
    TRAIN = std_TRAIN.copy()
    MODEL = std_MODEL.copy()

    TRAIN["hnm_period"] = 10

    config_name = f"emb{emb}_rat{ratio}_{model}_bi{bidir}_lay{layer}.json"

    DATASET['partition_data_name'] = config_name

    if emb == "25":
        MODEL["hidden_size"] = 25
    elif emb == "50":
        MODEL["hidden_size"] = 50
    elif emb == "100":
        MODEL["hidden_size"] = 100

    n_pos = 25000

    DATASET["initial_random_negatives"] = int(0.5 * n_pos / 5)
    DATASET["initial_jeremy_negatives"] = int(0.5 * n_pos / 5)

    DATASET["test_random_negatives"] = int(0.5 * n_pos / 5)
    DATASET["test_jeremy_negatives"] = int(0.5 * n_pos / 5)

    TRAIN["active"] = True
    TRAIN["hard_neg_cap"] = int(0.6 * 0.5 * n_pos)
    DATASET['random_mutability'] = 0.5
    DATASET['jeremy_mutability'] = 0.5  

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

    TRAIN["A_name"] = model
    MODEL["bidirectional"] = bool(bidir)

    if bidir == 1:
        MODEL["num_layers"] = layer


    out = {"DATASET_CONFIG": DATASET, "TRAIN_CONFIG": TRAIN, "MODEL_KWARGS": MODEL}
    config_path = os.path.join("configs", str(session_number), config_name)
    with open(config_path, 'w') as fp:
        json.dump(out, fp, indent=4)