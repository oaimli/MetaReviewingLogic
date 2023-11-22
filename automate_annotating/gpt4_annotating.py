import random
import openai
import time
import json


def parse_expression(result):
    judgements = []
    return judgements

def parse_facet(result):
    return ""

def parse_polarity(result):
    return ""

def parse_expressor(result):
    return ""

def parse_convincingness(result):
    return ""

def annotating(samples):
    """
    :param samples: dict
    :return: dict
    """
    prompt_expression = open("prompt_expression.txt").read()
    prompt_facet = open("prompt_facet.txt").read()
    prompt_expressor = open("prompt_expresser.txt").read()
    prompt_convincingness = open("prompt_convincingness.txt").read()
    prompt_polarity = open("prompt_polarity.txt").read()

    results = {}
    for sample in samples:
        print(sample["paper_id"])
        print(sample.keys())
        documents_annotated = []
        for source_document in sample["documents"]:
            # get content expression and sentiment expression
            while True:
                try:
                    output_dict = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": prompt_expression.replace("{{source_document}}",
                                                                                    source_document[
                                                                                        "document_content"])}
                        ]
                    )
                    judgements = parse_expression(output_dict['choices'][0]['message']['content'])
                    break
                except Exception as e:
                    print(e)
                    if ("limit" in str(e)):
                        time.sleep(2)

            for judgement in judgements:
                content_expression = judgement["Content Expression"]
                sentiment_expression = judgement["Sentiment Expression"]
                judgement_expression = content_expression + " " + sentiment_expression

                # get criteria facet
                while True:
                    try:
                        output_dict = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": prompt_facet.replace("{{source_document}}",
                                                                                   source_document[
                                                                                       "document_content"]).replace(
                                    "{{judgement_expression}}", judgement_expression)}
                            ]
                        )
                        judgement["Criteria Facet"] = parse_facet(output_dict['choices'][0]['message']['content'])
                        break
                    except Exception as e:
                        print(e)
                        if ("limit" in str(e)):
                            time.sleep(2)

                # get polarity
                while True:
                    try:
                        output_dict = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": prompt_polarity.replace("{{source_document}}",
                                                                                   source_document[
                                                                                       "document_content"]).replace(
                                    "{{judgement_expression}}", judgement_expression)}
                            ]
                        )
                        judgement["Sentiment Polarity"] = parse_polarity(
                            output_dict['choices'][0]['message']['content'])
                        break
                    except Exception as e:
                        print(e)
                        if ("limit" in str(e)):
                            time.sleep(2)

                # get expressor
                while True:
                    try:
                        output_dict = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": prompt_expressor.replace("{{source_document}}",
                                                                                      source_document[
                                                                                          "document_content"]).replace(
                                    "{{judgement_expression}}", judgement_expression)}
                            ]
                        )
                        judgement["Sentiment Expresser"] = parse_expressor(
                            output_dict['choices'][0]['message']['content'])
                        break
                    except Exception as e:
                        print(e)
                        if ("limit" in str(e)):
                            time.sleep(2)

                # get convincingness
                while True:
                    try:
                        output_dict = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system",
                                 "content": prompt_expressor.replace("{{source_document}}",
                                                                     source_document[
                                                                         "document_content"]).replace(
                                     "{{judgement_expression}}", judgement_expression)}
                            ]
                        )
                        judgement["Convincingness"] = parse_convincingness(
                            output_dict['choices'][0]['message']['content'])
                        break
                    except Exception as e:
                        print(e)
                        if ("limit" in str(e)):
                            time.sleep(2)

            documents_annotated.append(
                {"Document Title": source_document["document_title"], "Annotated Judgements": judgements})
        results[sample["paper_id"]] = documents_annotated

    return results


if __name__ == "__main__":
    random.seed(42)
    openai.api_key = "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"

    with open("../annotation_analysis/bryan_annotation_result.json") as f:
        bryan_results = json.load(f)
    with open("../annotation_analysis/zenan_annotation_result.json") as f:
        zenan_results = json.load(f)
    assert len(set(bryan_results.keys()).difference(set(zenan_results.keys()))) == 0
    samples_annotated_keys = bryan_results.keys()

    with open("gpt4_annotation_data_small.json") as f:
        samples_all = json.load(f)
    # Evaluation data for agreement of GPT-4 with human annotators
    samples_gpt4 = []
    for sample in samples_all:
        if sample["paper_id"] in samples_annotated_keys:
            samples_gpt4.append(sample)
    samples_gpt4 = random.sample(samples_gpt4, 1)
    results = annotating(samples_gpt4)
    with open("gpt4_annotation_result_small.json", "w") as f:
        json.dump(zenan_results, f, indent=4)

    # with open("gpt4_annotation_data_large.json") as f:
    #     samples_all = json.load(f)
    # # Annotating more data with GPT-4
    # samples_gpt4 = []
    # for sample in samples_all:
    #     if sample["paper_id"][10:] not in samples_annotated_keys and sample["label"] == "train":
    #         samples_gpt4.append(sample)
    # samples_gpt4 = random.sample(samples_gpt4, 2)
    # results = annotating(samples_gpt4)
    # with open("gpt4_annotation_result_large.json", "w") as f:
    #     json.dump(zenan_results, f, indent=4)
