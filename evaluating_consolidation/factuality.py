import json
import random

import numpy as np
from summac.model_summac import SummaCZS, SummaCConv

def summac_scores(documents, summaries):
    model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
    score_zs = model_zs.score(documents, summaries)
    score_conv = model_conv.score(documents, summaries)
    return score_zs["scores"], score_conv["scores"]

def evaluating(test_samples, generations):
    documents = []
    summaries = []
    for id in test_samples.keys():
        cluster = []
        for review in test_samples[id]["reviews"]:
            cluster.append(review["comment"])
        documents.append("\n".join(cluster))
        summaries.append(generations[id]["generation"])

    scores_zs, scores_conv = summac_scores(documents, summaries)
    print(np.mean(scores_zs), np.mean(scores_conv))



if __name__ == "__main__":
    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
        One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
        The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
        Arcadia Planitia is in Mars' northern lowlands."""
    summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
    scores_zs, scores_conv = summac_scores([document], [summary])
    print("scores zs", np.mean(scores_zs), "scores conv", np.mean(scores_conv))


    with open("../enhancing_prompting/test_data.json") as f:
        test_samples = json.load(f)
    print(len(test_samples))

    # ids_samples = test_samples.keys()
    # target_set = random.sample(ids_samples, 32)
    # tmp = {}
    # for id in target_set:
    #     tmp[id] = test_samples[id]
    # test_samples = tmp
    # print(len(test_samples))

    print("generation_gpt35_prompt_naive")
    with open("../enhancing_prompting/results/generation_gpt35_prompt_naive.json") as f:
        generations_prompt_naive = json.load(f)
    evaluating(test_samples, generations_prompt_naive)

    print("generation_gpt35_prompt_llm")
    with open("../enhancing_prompting/results/generation_gpt35_prompt_llm.json") as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    print("generation_gpt35_prompt_ours")
    with open("../enhancing_prompting/results/generation_gpt35_prompt_ours.json") as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)


    print("generation_gpt35_pipeline_ours")
    with open("../enhancing_prompting/results/generation_gpt35_pipeline_ours.json") as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_prompt_naive"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_prompt_llm"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_prompt_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_gpt4_pipeline_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_prompt_naive"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_prompt_llm"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_prompt_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_7b_pipeline_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_prompt_naive"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_prompt_llm"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_prompt_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)

    task = "generation_llama2_70b_pipeline_ours"
    print(task)
    with open("../enhancing_prompting/results/%s.json" % task) as f:
        generations_prompt_llm = json.load(f)
    evaluating(test_samples, generations_prompt_llm)




