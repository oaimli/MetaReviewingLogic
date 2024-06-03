import json
import random
import json


with open("../enhancing_prompting/test_data.json") as f:
    test_samples = json.load(f)
print(len(test_samples))

random.seed(42)
ids_samples = test_samples.keys()
target_set = random.sample(ids_samples, 30)
tmp = {}
for id in target_set:
    tmp[id] = test_samples[id]
test_samples = tmp
print(len(test_samples))


task = "generation_gpt4_prompt_ours"
print(task)
with open("../enhancing_prompting/results/%s.json" % task) as f:
    generations_prompt_ours_gpt4 = json.load(f)

cluster_1 = []
for test_sample_id in test_samples.keys():
    test_sample = test_samples[test_sample_id]
    tmp = {}
    tmp["paper_id"] = test_sample["paper_id"]
    meta_review_1 = {"human_written": test_sample["meta_review"]}
    meta_review_2 = {"gpt4_prompt_ours": generations_prompt_ours_gpt4[test_sample_id]["generation"]}
    meta_reviews = [meta_review_1, meta_review_2]
    random.shuffle(meta_reviews)
    tmp["meta_reviews"] = meta_reviews
    source_documents = []
    source_documents.append({"title": "Abstract", "content": test_sample["paper_abstract"]})
    for review in test_sample["reviews"]:
        source_documents.append({"title": review["title"], "content": review["comment"]})
    tmp["source_documents"] = source_documents
    cluster_1.append(tmp)

with open("gpt4_prompt_ours_vs_human_original.json", "w") as f:
    json.dump(cluster_1, f, indent=4)



task = "generation_llama2_70b_prompt_naive"
print(task)
with open("../enhancing_prompting/results/%s.json" % task) as f:
    generations_prompt_naive_llama2_70b = json.load(f)


task = "generation_llama2_70b_prompt_ours"
print(task)
with open("../enhancing_prompting/results/%s.json" % task) as f:
    generations_prompt_ours_llama2_70B = json.load(f)

cluster_2 = []
for test_sample_id in test_samples.keys():
    test_sample = test_samples[test_sample_id]
    tmp = {}
    tmp["paper_id"] = test_sample["paper_id"]
    meta_review_1 = {"llama2_70b_prompt_naive": generations_prompt_naive_llama2_70b[test_sample_id]["generation"]}
    meta_review_2 = {"llama2_70b_prompt_ours": generations_prompt_ours_llama2_70B[test_sample_id]["generation"]}
    meta_reviews = [meta_review_1, meta_review_2]
    random.shuffle(meta_reviews)
    tmp["meta_reviews"] = meta_reviews
    source_documents = []
    source_documents.append({"title": "Abstract", "content": test_sample["paper_abstract"]})
    for review in test_sample["reviews"]:
        source_documents.append({"title": review["title"], "content": review["comment"]})
    tmp["source_documents"] = source_documents
    cluster_2.append(tmp)

with open("llama2_70b_prompt_ours_vs_prompt_naive_original.json", "w") as f:
    json.dump(cluster_2, f, indent=4)