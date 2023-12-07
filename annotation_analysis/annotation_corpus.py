import numpy as np
import json


with open("../annotation_data/annotation_data_small.json") as f:
    samples = json.load(f)
print("annotation data", len(samples.keys()))

annotated_document_counts = []
lens_samples = []
lens_source_documents = []
lens_meta_review = []
for id, sample in samples.items():
    meta_review = sample["meta_review"]
    reviews = sample["reviews"]
    annotated_document_counts.append(len(reviews) + 1)

    source_documents = []
    for review in reviews:
        source_documents.append(review["comment"])
        lens_source_documents.append(len(review["comment"].split()))

    length_meta = len(meta_review.split())
    lens_meta_review.append(length_meta)
    length_source = len(" ".join(source_documents).split())
    lens_samples.append(length_source + length_meta)

print("document count to be annotated per sample", np.mean(annotated_document_counts), np.min(annotated_document_counts), np.max(annotated_document_counts))
print("length of samples", np.mean(lens_samples), np.min(lens_samples), np.max(lens_samples))
print("length of source documents", np.mean(lens_source_documents), np.min(lens_source_documents), np.max(lens_source_documents))
print("length of meta-reviews", np.mean(lens_meta_review), np.min(lens_meta_review), np.max(lens_meta_review))