import json
import os

target_folder = "gpt4_result_small"
files_all = os.listdir(target_folder)
results_all = {}
for file in files_all:
    results_all.update(json.load(open(os.path.join(target_folder, file))))

with open("gpt4_result_small.json", "w") as f:
    json.dump(results_all, f, indent=4)
