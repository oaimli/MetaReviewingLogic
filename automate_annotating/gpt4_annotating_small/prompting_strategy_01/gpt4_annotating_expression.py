import random
import jsonlines
import openai
import time
import json


def parse_expression(output):
    with open("tmp.jsonl", "w") as f:
        f.write(output.strip())
    results = []
    try:
        with jsonlines.open("tmp.jsonl") as reader:
            for line in reader:
                results.append(line)
    except jsonlines.InvalidLineError as err:
        print("Jsonlines parsing error,", err)
    return results


def get_result(prompt_format, document):
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt_format.replace("{{source_document}}", document)}
                ],
                n=5
            )
            results = []
            for output in output_dict['choices']:
                tmp = parse_expression(output['message']['content'])
                if len(tmp) > len(results):
                    results = tmp
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    judgements = []
    for line in results:
        if "content_expression" in line.keys() and "sentiment_expression" in line.keys():
            if line["content_expression"].strip() != "" and line["sentiment_expression"].strip() != "":
                judgements.append(
                    {"Content Expression": line["content_expression"], "Sentiment Expression": line["sentiment_expression"],
                     "Criteria Facet": "",
                     "Sentiment Polarity": "", "Sentiment Expresser": "", "Convincingness": ""})
    print(judgements)
    return judgements



def annotating(samples):
    """
    :param samples: dict
    :return: none
    """
    print("Annotating count", len(samples))
    prompt_expression = open("prompts/prompt_expression.txt").read()

    for paper_id, sample in samples.items():
        print(paper_id)
        print(sample.keys())

        documents_annotated = []
        # annotating the meta-review
        judgements = get_result(prompt_expression, sample["meta_review"])
        if len(judgements) > 0:
            documents_annotated.append(
                {"Document Title": sample["meta_review_title"], "Annotated Judgements": judgements})

        for review in sample["reviews"]:
            # get content expression and sentiment expression
            judgements = get_result(prompt_expression, review["comment"])
            if len(judgements) > 0:
                documents_annotated.append(
                    {"Document Title": review["title"], "Annotated Judgements": judgements})
        results = {}
        results[paper_id] = documents_annotated
        with open("../gpt4_result_small/%s.json" % paper_id, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    random.seed(42)
    openai.api_key = "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"

    with open("../../../annotation_data/annotation_data_small.json") as f:
        samples_all = json.load(f)

    f = open("experiment_ids_dev.txt")
    ids = f.read().split("\n")

    # Evaluation data for agreement of GPT-4 with human annotators
    samples_gpt4 = {}
    for sample_key in samples_all.keys():
        if sample_key in ids:
            samples_gpt4[sample_key] = samples_all[sample_key]
    annotating(samples_gpt4)
