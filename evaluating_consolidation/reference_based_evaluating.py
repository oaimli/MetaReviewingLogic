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
    # Load the annotated judgements
    for sample in os.listdir("judgements/test_data"):
        with open(os.path.join("judgements/test_data", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        test_samples[sample[:-5]]["gpt4_judgements"] = judgements

    with open("../enhancing_prompting/results/generation_gpt35_prompt_naive.json") as f:
        generations_prompt_naive = json.load(f)
    # Load the annotated judgements
    for sample in os.listdir("judgements/generation_gpt35_prompt_naive"):
        with open(os.path.join("judgements/generation_gpt35_prompt_naive", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_naive[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_naive)

    with open("../enhancing_prompting/results/generation_gpt35_prompt_llm.json") as f:
        generations_prompt_llm = json.load(f)
    # Load the annotated judgements
    for sample in os.listdir("judgements/generation_gpt35_prompt_llm"):
        with open(os.path.join("judgements/generation_gpt35_prompt_llm", sample)) as f:
            judgements = json.load(f)[sample[:-5]]
        generations_prompt_llm[sample[:-5]]["gpt4_judgements"] = judgements
    evaluating(test_samples, generations_prompt_llm)

