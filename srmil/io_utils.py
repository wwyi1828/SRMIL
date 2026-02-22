import json
import os


def update_results(json_path, dataset_name, baseline_name, accuracy, auc_list):
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            results = json.load(file)
    else:
        results = {}

    if dataset_name not in results:
        results[dataset_name] = {}
    if baseline_name not in results[dataset_name]:
        results[dataset_name][baseline_name] = {"accuracy": []}
        for index in range(len(auc_list)):
            results[dataset_name][baseline_name].update({f"auc_{index}": []})

    results[dataset_name][baseline_name]["accuracy"].append(accuracy)
    for index, auc_score in enumerate(auc_list):
        auc_key = f"auc_{index}"
        results[dataset_name][baseline_name][auc_key].append(auc_score)

    with open(json_path, "w") as file:
        json.dump(results, file, indent=4)
