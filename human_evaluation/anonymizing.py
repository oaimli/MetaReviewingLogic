import json

with open("gpt4_prompt_ours_vs_human_original.json", "r") as f:
    original = json.load(f)

annotations = []
for sample in original:
    sample_new = {}
    meta_reviews = sample["meta_reviews"]
    meta_reviews_anonymized = []
    meta_review_a = meta_reviews[0]
    meta_reviews_anonymized.append({"A": list(meta_review_a.values())[0]})
    meta_review_b = meta_reviews[1]
    meta_reviews_anonymized.append({"B": list(meta_review_b.values())[0]})
    sample_new["paper_id"] = sample["paper_id"]
    sample_new["meta_reviews"] = meta_reviews_anonymized
    sample_new["your_preference"] = ""
    sample_new["source_documents"] = sample["source_documents"]
    annotations.append(sample_new)

with open("gpt4_prompt_ours_vs_human_annotation.json", "w") as f:
    json.dump(annotations, f, indent=4)




with open("llama2_70b_prompt_ours_vs_prompt_naive_original.json", "r") as f:
    original = json.load(f)

annotations = []
for sample in original:
    sample_new = {}
    meta_reviews = sample["meta_reviews"]
    meta_reviews_anonymized = []
    meta_review_a = meta_reviews[0]
    meta_reviews_anonymized.append({"A": list(meta_review_a.values())[0]})
    meta_review_b = meta_reviews[1]
    meta_reviews_anonymized.append({"B": list(meta_review_b.values())[0]})
    sample_new["paper_id"] = sample["paper_id"]
    sample_new["meta_reviews"] = meta_reviews_anonymized
    sample_new["your_preference"] = ""
    sample_new["source_documents"] = sample["source_documents"]
    annotations.append(sample_new)

with open("llama2_70b_prompt_ours_vs_prompt_naive_annotation.json", "w") as f:
    json.dump(annotations, f, indent=4)