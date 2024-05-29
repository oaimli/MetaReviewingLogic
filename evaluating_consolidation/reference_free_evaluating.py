import openai
import json
import os
import sys
sys.path.append("../")
from evaluating_consolidation.fusion_eval.fusion_eval import annotating_with_judgements

openai.api_key = "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"

# Load meta-review judgements for a specific model, the judgements here are borrowed from facet-eval
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt4_prompt_naive"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt4_prompt_llm"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt4_prompt_ours"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt4_pipeline_ours"

# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_7b_prompt_naive"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_7b_prompt_llm"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_7b_prompt_ours"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_7b_pipeline_ours"

# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt35_prompt_naive"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt35_prompt_llm"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt35_prompt_ours"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_gpt35_pipeline_ours"

judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_70b_prompt_naive"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_70b_prompt_llm"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_70b_prompt_ours"
# judgements_folder_generated_meta_review = "facet_eval_judgements_tmp/generation_llama2_70b_pipeline_ours"

#
print("########", judgements_folder_generated_meta_review)

meta_review_judgements_all = {}
for sample in os.listdir(judgements_folder_generated_meta_review):
    with open(os.path.join(judgements_folder_generated_meta_review, sample)) as f:
        judgements = json.load(f)[sample[:-5]]
    meta_review_judgements_all[sample[:-5]] = judgements

# Load source judgements, all models share the same
judgements_folder_source = "fusion_eval_judgements_tmp/test_data"
source_judgements_all = {}
for sample in os.listdir(judgements_folder_source):
    with open(os.path.join(judgements_folder_source, sample)) as f:
        judgements = json.load(f)[sample[:-5]]
    source_judgements_all[sample[:-5]] = judgements

print("samples", len(meta_review_judgements_all.keys()), len(source_judgements_all.keys()))
shared_keys = set(meta_review_judgements_all).intersection(set(source_judgements_all))

instances = []
for key in shared_keys:
    meta_review_judgements = meta_review_judgements_all[key]
    source_judgements = source_judgements_all[key]

    for meta_review_judgement in meta_review_judgements:
        source_judgements_facet = []
        for judgement in source_judgements:
            if judgement["Criteria Facet"] == meta_review_judgement["Criteria Facet"]:
                source_judgements_facet.append(judgement)
        instance = {}
        instance["meta_review_judgement"] = meta_review_judgement
        instance["source_judgements"] = source_judgements_facet
        instances.append(instance)
print("instances", len(instances))

# facets = ["Advancement", "Soundness", "Novelty", "Overall", "Clarity", "Compliance"]
facets = ["Clarity"]
correct_all = 0
for facet in facets:
    print(facet)
    instances_facet = []
    for instance in instances:
        judgement = instance["meta_review_judgement"]
        if judgement["Criteria Facet"] == facet:
            instances_facet.append(instance)
    print("Judgements in this facet", len(instances_facet))

    correct = 0
    for instance in instances_facet:
        correct += annotating_with_judgements(instance["source_judgements"], instance["meta_review_judgement"], prompt_file="fusion_eval/prompt_for_judgements.txt")

    print("Correct in %s" % facet, correct, correct / len(instances_facet))

    correct_all += correct

print("Correct in all", correct_all, correct_all / len(instances))