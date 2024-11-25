import json

from sklearn.metrics import cohen_kappa_score
import krippendorff
import numpy as np


# competitors = "gpt4_prompt_ours_vs_human"
competitors = "llama2_70b_prompt_ours_vs_prompt_naive"

with open(f"{competitors}_original.json", "r") as f:
    original = json.load(f)

with open(f"{competitors}_annotation_1.json", "r") as f:
    annotations_1 = json.load(f)

with open(f"{competitors}_annotation_2.json", "r") as f:
    annotations_2 = json.load(f)

with open(f"{competitors}_annotation_3.json", "r") as f:
    annotations_3 = json.load(f)

comparisons = {}
results_1 = []
results_2 = []
results_3 = []
for sample, annotation_1, annotation_2, annotation_3 in zip(original, annotations_1, annotations_2, annotations_3):
    paper_id = sample["paper_id"]
    meta_review_generators = [list(sample["meta_reviews"][0].keys())[0], list(sample["meta_reviews"][1].keys())[0]]
    preference_1 = annotation_1["your_preference"]
    if "A" in preference_1:
        preference_1_no = 0
    elif "B" in preference_1:
        preference_1_no = 1
    else:
        print(f"The annotation is not correct for {paper_id} in annotation_1")
        preference_1_no = -1
    results_1.append(preference_1_no)

    preference_2 = annotation_2["your_preference"]
    if "A" in preference_2:
        preference_2_no = 0
    elif "B" in preference_2:
        preference_2_no = 1
    else:
        print(f"The annotation is not correct for {paper_id} in annotation_2")
        preference_2_no = -1
    results_2.append(preference_2_no)

    preference_3 = annotation_3["your_preference"]
    if "A" in preference_3:
        preference_3_no = 0
    elif "B" in preference_3:
        preference_3_no = 1
    else:
        print(f"The annotation is not correct for {paper_id} in annotation_3")
        preference_3_no = -1
    results_3.append(preference_3_no)

    # majority voting
    preferences = [preference_1_no, preference_2_no, preference_3_no]
    print(preferences)
    final_preference = max(preferences, key=preferences.count)
    print(final_preference)
    if final_preference == -1:
        print(paper_id)
    generator_name = meta_review_generators[final_preference]
    comparisons[generator_name] = comparisons.get(generator_name, 0) + 1
    print(comparisons)

print(comparisons)

# IAA
numerical_1 = []
numerical_2 = []
numerical_3 = []
equal_count_all = 0
equal_count_1v2 = 0
equal_count_1v3 = 0
equal_count_2v3 = 0
for item_1, item_2, item_3 in zip(numerical_1, numerical_2, numerical_3):
    # print(item_1, item_2, item_3)
    if item_1 == item_2 == item_3:
        equal_count_all += 1
    if item_1 == item_2:
        equal_count_1v2 += 1
    if item_1 == item_3:
        equal_count_1v3 += 1
    if item_2 == item_3:
        equal_count_2v3 += 1
print(numerical_1)
print(numerical_2)
print(numerical_3)
print("Equal rate all", equal_count_all / len(numerical_1))
print("Equal rate 1v2", equal_count_1v2 / len(numerical_1))
print("Equal rate 1v3", equal_count_1v3 / len(numerical_1))
print("Equal rate 2v3", equal_count_2v3 / len(numerical_1))
cohen_kappas = []
tmp = [results_1, results_2, results_3]
for i, results_i in enumerate(tmp):
    for j, results_j in enumerate(tmp):
        if j>i:
            cohen_kappas.append(cohen_kappa_score(results_i, results_j))
print(np.average(cohen_kappas))

print(krippendorff.alpha(reliability_data=np.array([results_1, results_2, results_3]), level_of_measurement="nominal"))



