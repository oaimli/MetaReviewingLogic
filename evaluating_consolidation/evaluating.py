import json
from rouge import rouge
import pandas as pd
from facet_eval import facet_eval


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

        all_results.append(result)
    all_results = pd.DataFrame(all_results)
    print(all_results.mean(axis=0))


if __name__ == "__main__":
    with open("../enhancing_prompting/test_data.json") as f:
        test_samples = json.load(f)

    with open("../enhancing_prompting/results/generation_gpt35_prompt_naive.json") as f:
        generations_prompt_naive = json.load(f)
    evaluating(test_samples, generations_prompt_naive)

    with open("../enhancing_prompting/results/generation_gpt35_prompt_llm.json") as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

