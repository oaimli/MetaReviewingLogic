import json
import os


with open("../enhancing_prompting/test_data.json") as f:
    test_data = json.load(f)
print(len(test_data))

with open("../enhancing_prompting/results/generation_gpt4_prompt_naive.json") as f:
    results_gpt4_prompt_naive = json.load(f)
print(len(results_gpt4_prompt_naive))

with open("../enhancing_prompting/results/generation_gpt4_prompt_ours.json") as f:
    results_gpt4_prompt_ours = json.load(f)
print(len(results_gpt4_prompt_ours))

test_judgements = {}
test_judgements_folder ="../evaluating_consolidation/facet_eval_judgements_tmp/test_data"
for file in os.listdir(test_judgements_folder):
    with open(os.path.join(test_judgements_folder, file)) as f:
        test_judgements.update(json.load(f))

gpt4_prompt_naive_judgements = {}
gpt4_prompt_naive_judgements_folder ="../evaluating_consolidation/facet_eval_judgements_tmp/generation_gpt4_prompt_naive"
for file in os.listdir(gpt4_prompt_naive_judgements_folder):
    with open(os.path.join(gpt4_prompt_naive_judgements_folder, file)) as f:
        gpt4_prompt_naive_judgements.update(json.load(f))

gpt4_prompt_ours_judgements = {}
gpt4_prompt_ours_judgements_folder ="../evaluating_consolidation/facet_eval_judgements_tmp/generation_gpt4_prompt_ours"
for file in os.listdir(gpt4_prompt_ours_judgements_folder):
    with open(os.path.join(gpt4_prompt_ours_judgements_folder, file)) as f:
        gpt4_prompt_ours_judgements.update(json.load(f))

output_pairs = []
for key in test_data:
    original_sample = test_data[key]
    largest_score = -1
    maximum_gap = 0
    for review in original_sample["reviews"]:
        rating = review["rating"]
        if rating > largest_score:
            largest_score = rating
        for review_other in original_sample["reviews"]:
            rating_other = review_other["rating"]
            if rating >0 and rating_other>0:
                gap = abs(rating_other - rating)
                if gap > maximum_gap:
                    maximum_gap = gap
    if maximum_gap>4 and largest_score>=8:
        print(key)
        print(maximum_gap, largest_score)
        sample = {}
        sample["paper_id"] = key
        sample["meta_review_human_written"] = test_data[key]["meta_review"]
        sample["meta_review_gpt4_prompt_naive"] = results_gpt4_prompt_naive[key]["generation"]
        sample["meta_review_gpt4_prompt_ours"] = results_gpt4_prompt_ours[key]["generation"]
        sample["judgements_human_written"] = test_judgements[key]
        sample["judgements_gpt4_prompt_naive"] = gpt4_prompt_naive_judgements[key]
        sample["judgements_gpt4_prompt_ours"] = gpt4_prompt_ours_judgements[key]
        output_pairs.append(sample)

print(len(output_pairs))
with open("comparison.json", "w") as f:
    json.dump(output_pairs, f, indent=4)



