import random
import jsonlines
import openai
import time
import json
from tqdm import tqdm


def predicting_prompt_llm(input_text):
    # generating steps
    prompt_format_steps = open("prompts/prompt_generating_steps.txt").read()
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system",
                     "content": prompt_format_steps.replace("{{input_documents}}", input_text)}
                ],
                n=1
            )
            generated_steps = output_dict['choices'][0]['message']['content']
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)

    # generate the meta-review
    prompt_format_metareview = open("prompts/prompt_llm.txt").read()
    # print(generated_steps)
    # print(prompt_format_metareview.replace("{{input_documents}}", input_text).replace("{{generated_steps}}", generated_steps))
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system",
                     "content": prompt_format_metareview.replace("{{input_documents}}", input_text).replace("{{generated_steps}}", generated_steps)}
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

    results = {}
    for key, sample in tqdm(test_samples.items()):
        input_texts = []
        for review in sample["reviews"]:
            if review["writer"] == "official_reviewer":
                input_texts.append(review["comment"])
        result = predicting_prompt_llm("\n".join(input_texts))
        results[key] = {"generation": result}
        # break

    print(len(results))
    with open("results/generation_gpt4_prompt_llm.json", "w") as f:
        json.dump(results, f, indent=4)