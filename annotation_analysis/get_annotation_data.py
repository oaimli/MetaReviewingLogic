import random

import jsonlines
import json

with open("bryan_annotation_result.json") as f:
    bryan_results = json.load(f)
with open("zenan_annotation_result.json") as f:
    zenan_results = json.load(f)
assert len(set(bryan_results.keys()).difference(set(zenan_results.keys()))) == 0
samples_annotated_keys = bryan_results.keys()


tmp = []
with jsonlines.open("../automate_annotating/peersum_all.jsonl") as reader:
    for line in reader:
        paper_id = line["paper_id"][10:]
        if paper_id in samples_annotated_keys:
            tmp.append(line)

samples = {}
for line in random.sample(tmp, 5):
    paper_id = line["paper_id"][10:]
    documents_new = []
    documents_new.append({"review_id": "meta-review", "writer": "meta-review", "comment": line["meta_review"], "rating": "-1", "confidence": "-1", "reply_to": "-1"})
    for doc in line["reviews"]:
        doc["document_title"] = ""
        documents_new.append(doc)

    line["documents"] = documents_new
    del line["meta_review"]
    del line["reviews"]
    samples[paper_id] = line

    # paper_title: str
    # paper_abstract, str
    # paper_acceptance, str
    # meta_review, str
    # reviews, [{review_id, writer, comment, rating, confidence, reply_to}] (all reviews and comments)
    # label, str, (train, val, test)

print("All annotation data", len(samples))

with open("annotation_data.json", "w") as f:
    json.dump(samples, f, indent=4)

