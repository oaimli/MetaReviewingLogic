import random

import jsonlines
import json

with open("../annotation_analysis/bryan_annotation_result.json") as f:
    bryan_results = json.load(f)
with open("../annotation_analysis/zenan_annotation_result.json") as f:
    zenan_results = json.load(f)
assert len(set(bryan_results.keys()).difference(set(zenan_results.keys()))) == 0
samples_annotated_keys = bryan_results.keys()

annotation_folder = "../../HumanAnnotation/mrg_judgement"
samples = {}
tmp = []
with jsonlines.open(annotation_folder + "/annotation/sampled_data.jsonl") as reader:
    for line in reader:
        tmp.append(line)
for line in random.sample(tmp, 5):
    paper_id = line["paper_id"][10:]
    if paper_id in samples_annotated_keys:
        source_documents_new = []
        source_documents_new.append({"document_title": "", "document_content": line["summary"]})
        for doc in line["source_documents"]:
            source_documents_new.append({"document_title": "", "document_content": doc})

        line["documents"] = source_documents_new
        del line["paper_id"]
        del line["summary"]
        del line["source_documents"]
        samples[paper_id] = line

print("All annotated count", len(samples))

with open("gpt4_annotation_data_small.json", "w") as f:
    json.dump(samples, f, indent=4)

