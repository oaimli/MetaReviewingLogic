import json
with open("results/generation_gpt35_prompt_llm.json") as f:
    results = json.load(f)

results_new = {}
for key in results:
    results_new[key] = {"generation": results[key]}

with open("results/generation_gpt35_prompt_llm.json", "w") as f:
    json.dump(results_new, f, indent=4)