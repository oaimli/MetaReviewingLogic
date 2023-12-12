import json
import os

f = open("experiment_ids.txt")
ids = f.read().split("\n")
print(ids)


target_folder = "gpt4_result_small"
files_all = os.listdir(target_folder)
results_all = {}
for file in files_all:
    if file[:-5] in ids:
        results_all.update(json.load(open(os.path.join(target_folder, file))))

with open("gpt4_result_small.json", "w") as f:
    json.dump(results_all, f, indent=4)
