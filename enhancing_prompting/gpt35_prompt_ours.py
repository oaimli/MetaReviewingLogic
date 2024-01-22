import random
import jsonlines
import openai
import time
import json


def predicting_prompt_naive(input_text):
    prompt_format = open("prompts/prompt_naive.txt").read()
    print(prompt_format)
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system",
                     "content": prompt_format.replace("{{input_documents}}", input_text)}
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
    for key, sample in test_samples.items():
        input_texts = []
        for review in sample["reviews"]:
            input_texts.append(review["comment"])
        result = predicting_prompt_naive("\n".join(input_texts))
        results[key] = {"generation": result}

    print(len(results))
    with open("results/generation_gpt35_prompt_naive.json", "w") as f:
        json.dump(results, f, indent=4)