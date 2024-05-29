import json
import os

from rouge import rouge
import pandas as pd
from facet_eval.facet_eval import facet_score


def evaluating(test_samples, generations):
    all_results = []
    for key, sample in test_samples.items():
        ground_truth = sample['meta_review']
        generation = generations[key]["generation"]
        result = {}
        # rouge
        rouge_result = rouge(ground_truth, generation)
        result.update(rouge_result)
        # facet-eval
        judgements_reference = sample["gpt4_judgements"]
        judgements_candidate = generations[key]["gpt4_judgements"]
        # print(judgements_candidate)
        # print(judgements_reference)
        facet_result = facet_score(judgements_reference, judgements_candidate)
        # print(facet_result)
        result.update(facet_result)
        all_results.append(result)
    all_results = pd.DataFrame(all_results)
    print(all_results.mean(axis=0))


if __name__ == "__main__":

    with open("../enhancing_prompting/test_data.json") as f:
        test_samples = json.load(f)
    # Load the annotated judgements for meta-reviews of test samples
    for sample in os.listdir("facet_eval_judgements_tmp/test_data"):
        with open(os.path.join("facet_eval_judgements_tmp/test_data", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        test_samples[sample[:-5]]["gpt4_judgements"] = judgements

    print("generation_gpt35_prompt_naive")
    with open("../enhancing_prompting/results/generation_gpt35_prompt_naive.json") as f:
        generations_prompt_naive = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/generation_gpt35_prompt_naive"):
        with open(os.path.join("facet_eval_judgements_tmp/generation_gpt35_prompt_naive", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_naive[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_naive)

    print("generation_gpt35_prompt_llm")
    with open("../enhancing_prompting/results/generation_gpt35_prompt_llm.json") as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/generation_gpt35_prompt_llm"):
        with open(os.path.join("facet_eval_judgements_tmp/generation_gpt35_prompt_llm", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    print("generation_gpt35_prompt_ours")
    with open("../enhancing_prompting/results/generation_gpt35_prompt_ours.json") as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/generation_gpt35_prompt_ours"):
        with open(os.path.join("facet_eval_judgements_tmp/generation_gpt35_prompt_ours", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)


    print("generation_gpt35_pipeline_ours")
    with open("../enhancing_prompting/results/generation_gpt35_pipeline_ours.json") as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/generation_gpt35_pipeline_ours"):
        with open(os.path.join("facet_eval_judgements_tmp/generation_gpt35_pipeline_ours", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_prompt_naive"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_prompt_llm"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_prompt_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_pipeline_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_prompt_naive"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_prompt_llm"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            tmp = json.load(f)
            judgements = tmp[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_prompt_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_pipeline_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_prompt_naive"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_prompt_llm"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_prompt_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_pipeline_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements of generated meta-reviews
    for sample in os.listdir("facet_eval_judgements_tmp/%s" % task):
        with open(os.path.join("facet_eval_judgements_tmp/%s" % task, sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)
