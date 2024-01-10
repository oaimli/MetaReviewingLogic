import random
import jsonlines
import openai
import time
import json


def annotating(pairs):
    print("Annotating count", len(pairs))
    prompt_format = open("prompt.txt").read()
    print(prompt_format)

    results = []
    for pair in pairs:
        judgement_i = pair[0]["Content Expression"] + " " + pair[0]["Sentiment Expression"]
        judgement_j = pair[1]["Content Expression"] + " " + pair[1]["Sentiment Expression"]
        print(judgement_i)
        print(judgement_j)
        while True:
            try:
                output_dict = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system",
                         "content": "what is your age"}
                    ],
                    n=5
                )
                tmp = []
                for output in output_dict['choices']:
                    output_content = output['message']['content'].lower()
                    print(output_content)
                    if "yes" in output_content:
                        tmp.append(1)
                    if "no" in output_content:
                        tmp.append(0)
                if sum(tmp) >= 3:
                    results.append(1)
                else:
                    results.append(0)
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
    return results


if __name__ == "__main__":
    openai.api_key = "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"

    with open("../annotation_analysis/bryan_annotation_result.json") as f:
        bryan_annotations = json.load(f)

    with open("../annotation_analysis/zenan_annotation_result.json") as f:
        zenan_annotations = json.load(f)

    shared_keys = set(bryan_annotations).intersection(set(zenan_annotations))

    pairs = []
    for key in shared_keys:
        bryan_annotation = bryan_annotations[key]
        zenan_annotation = zenan_annotations[key]
        judgements = []
        for document in bryan_annotation:
            judgements.extend(document["Annotated Judgements"])
        for document in zenan_annotation:
            judgements.extend(document["Annotated Judgements"])

        for i, judgement_i in enumerate(judgements):
            for j, judgement_j in enumerate(judgements):
                if j > i and judgement_i["Criteria Facet"] == judgement_j["Criteria Facet"]:
                    pairs.append([judgement_i, judgement_j])

    sample_pairs = random.sample(pairs, 10)
    results = annotating(sample_pairs)
    for pair, result in zip(sample_pairs, results):
        print(pair)
        print(result)
