import random

import jsonlines
import json

with open("../annotation_analysis/bryan_annotation_result.json") as f:
    bryan_results = json.load(f)
with open("../annotation_analysis/zenan_annotation_result.json") as f:
    zenan_results = json.load(f)
assert len(set(bryan_results.keys()).difference(set(zenan_results.keys()))) == 0
samples_annotated_keys = bryan_results.keys()

tmp = []
with jsonlines.open("peersum_all.json") as reader:
    for line in reader:
        paper_id = line["paper_id"][10:]
        if paper_id in samples_annotated_keys:
            tmp.append(line)

samples = {}
for line in random.sample(tmp, len(tmp)):
    paper_id = line["paper_id"][10:]
    samples[paper_id] = line

print("All annotation data", len(samples))

# Be careful, the data needs to manually add and update title in some samples
# with open("annotation_data_small_tmp.json", "w") as f:
#     json.dump(samples, f, indent=4)