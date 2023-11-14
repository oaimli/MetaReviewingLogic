import numpy as np
import jsonlines
import json
import pandas

annotation_folder = "../../HumanAnnotation/mrg_judgement"

samples = []
with jsonlines.open(annotation_folder + "/annotation/sampled_data.jsonl") as reader:
    for line in reader:
        samples.append(line)

with open("bryan_annotation_result.json") as f:
    annotated_ids = json.load(f).keys()

annotated_document_counts = []
lens_samples = []
lens_source_documents = []
lens_meta_review = []
contradicts = []
for sample in samples:
    id = sample["paper_id"][10:]
    print(id)
    if id in annotated_ids:
        source_documents = sample["source_documents"]
        meta_review = sample["summary"]
        contradict = sample["contradict"]
        annotated_document_counts.append(len(source_documents) + 1)
        contradicts.append(contradict)

        for source_document in source_documents:
            lens_source_documents.append(len(source_document.split()))

        length_meta = len(meta_review.split())
        lens_meta_review.append(length_meta)
        length_source = len(" ".join(source_documents).split())
        lens_samples.append(length_source + length_meta)

print("annotated_document_counts", np.mean(annotated_document_counts), np.min(annotated_document_counts), np.max(annotated_document_counts))
print("lens_samples", np.mean(lens_samples), np.min(lens_samples), np.max(lens_samples))
print("lens_source_documents", np.mean(lens_source_documents), np.min(lens_source_documents), np.max(lens_source_documents))
print("lens_meta_review", np.mean(lens_meta_review), np.min(lens_meta_review), np.max(lens_meta_review))

print("contradicts", pandas.value_counts(contradicts))