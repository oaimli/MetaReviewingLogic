import random
import jsonlines
import json

with open("../annotation_data/annotation_data_small.json") as f:
    annotation_data = json.load(f)

all_test_samples = {}
all_samples = {}
with jsonlines.open("../annotation_data/peersum_all.json") as reader:
    for line in reader:
        paper_id = line["paper_id"][10:]
        all_samples[paper_id] = line
        if line["label"] == "test":
            all_test_samples[paper_id] = line

all_keys = all_test_samples.keys()
annotation_keys = annotation_data.keys()
sampled_keys = []
for key in all_keys:
    if key not in annotation_keys:
        sampled_keys.append(key)
sampled_keys = random.sample(sampled_keys, 70)

sampled_data = {}
for key in sampled_keys:
    sampled_data[key] = all_test_samples[key]
for key in annotation_keys:
    sampled_data[key] = all_samples[key]

print("all original test data", len(all_test_samples))
print("sampled data", len(sampled_data))
with open("test_data.json", "w") as f:
    json.dump(sampled_data, f, indent=4)