import os

import openai
import time
import json
from tqdm import tqdm

def summarizing_judgements(criteria_facet, source_judgements):
    source_judgements_text = []
    for source_judgement in source_judgements:
        source_judgements_text.append(str(source_judgement))
    prompt_format = open("prompts/prompt_summarizing_each_facet.txt").read()
    prompt_format = prompt_format.replace("{{input_judgements}}", "\n".join(source_judgements_text)).replace("{{criteria_facet}}", criteria_facet)
    # print(prompt_format)
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system",
                     "content": prompt_format}
                ],
                n=1
            )
            output = output_dict['choices'][0]['message']['content']
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    return output


def aggregating_sub_summaries(input_text):
    prompt_format = open("prompts/prompt_aggregating_sub_summaries.txt").read()
    prompt_format = prompt_format.replace("{{input_sub_summaries}}", input_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system",
                     "content": prompt_format}
                ],
                n=1
            )
            output = output_dict['choices'][0]['message']['content']
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    return output


if __name__ == "__main__":
    openai.api_key = "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"

    with open("test_data.json") as f:
        test_samples = json.load(f)

    source_judgements = {}
    source_judgements_folder = "../evaluating_consolidation/fusion_eval_judgements_tmp/test_data"
    file_names = os.listdir(source_judgements_folder)
    for file_name in file_names:
        with open(os.path.join(source_judgements_folder, file_name)) as f:
            source_judgements.update(json.load(f))
    print(len(source_judgements))

    results = {}
    for key, sample in tqdm(test_samples.items()):
        judgements = source_judgements[key]
        organized = {}
        for judgement in judgements:
            criteria_facet = judgement["Criteria Facet"]
            tmp = organized.get(criteria_facet, [])
            tmp.append(judgement)
            organized[criteria_facet] = tmp

        sub_summaries = []
        for k, v in organized.items():
            sub_summary = summarizing_judgements(k, v)
            sub_summaries.append(k + "\n" + sub_summary)

        result = aggregating_sub_summaries("\n".join(sub_summaries))
        results[key] = {"generation": result}
        # break

    print(len(results))
    with open("results/generation_gpt4_pipeline_ours.json", "w") as f:
        json.dump(results, f, indent=4)