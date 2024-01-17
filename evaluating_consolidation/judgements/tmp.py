import json


with open("../../enhancing_prompting/test_data.json") as f:
    test_samples = json.load(f)

for key in test_samples:
    judgements = test_samples[key]["gpt4_judgements"]
    tmp = {key: judgements}
    with open("test_data/%s.json" % key, "w") as f:
        json.dump(tmp, f, indent=4)

with open("../../enhancing_prompting/results/generation_gpt35_prompt_naive.json") as f:
    generations_gpt35_prompt_naive = json.load(f)

for key in generations_gpt35_prompt_naive:
    print(key)
    print(generations_gpt35_prompt_naive[key].keys())
    judgements = generations_gpt35_prompt_naive[key]["gpt4_judgements"]
    tmp = {key: judgements}
    with open("generation_gpt35_prompt_naive/%s.json" % key, "w") as f:
        json.dump(tmp, f, indent=4)