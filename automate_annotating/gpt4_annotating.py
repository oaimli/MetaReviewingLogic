import random
import openai
import time
import json


def parse_expression(result):
    print(result)
    lines = result.split("\n")
    judgements = []
    for line in lines:
        items = line.split("-")
        judgements.append({"Content Expression": items[0], "Sentiment Expression": items[-1]})
    return judgements

def parse_facet(result):
    print(result)
    result = result.lower()
    if "novelty" in result:
        return "Novelty"
    elif "soundness" in result:
        return "Soundness"
    elif "clarity" in result:
        return "Clarity"
    elif "advancement" in result:
        return "Advancement"
    elif "compliance" in result:
        return "Compliance"
    elif "overall" in result:
        print("Cannot parse the result correctly")
        return "Overall"
    else:
        return "Overall"

def parse_polarity(result):
    print(result)
    if "strong" in result.lower() and "negative" in result.lower():
        return "Strong negative"
    elif "strong" not in result.lower() and "negative" in result.lower():
        return "Negative"
    elif "strong" in result.lower() and "positive" in result.lower():
        return "Strong positive"
    elif "strong" not in result.lower() and "positive" in result.lower():
        return "Positive"
    else:
        print("Cannot parse the result correctly")
        return "Negative"

def parse_expressor(result):
    print(result)
    result = result.lower()
    if "self" in result:
        return "Self"
    elif "others" in result:
        return "Others"
    else:
        print("Cannot parse the result correctly")
        return "Self"

def parse_convincingness(result):
    print(result)
    result = result.lower()
    if "not" in result and "applicable" in result:
        return "Not applicable"
    elif "not" in result and "at all" in result:
        return "Not at all"
    elif "slightly" in result and "convincing" in result:
        return "Slightly Convincing"
    elif "highly" in result and "convincing" in result:
        return "Highly Convincing"
    else:
        print("Cannot parse the result correctly")
        return "Slightly Convincing"

def annotating(samples):
    """
    :param samples: dict
    :return: none
    """
    print("Annotating count", len(samples))
    prompt_expression = open("prompt_expression.txt").read()
    prompt_facet = open("prompt_facet.txt").read()
    prompt_expressor = open("prompt_expresser.txt").read()
    prompt_convincingness = open("prompt_convincingness.txt").read()
    prompt_polarity = open("prompt_polarity.txt").read()

    for paper_id, sample in samples.items():
        print(paper_id)
        print(sample.keys())
        documents_annotated = []
        for source_document in sample["documents"]:
            if source_document["document_title"] == "Abstract":
                continue
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
                if judgement["Sentiment Expresser"] == "Others":
                    judgement["Convincingness"] = "Not applicable"
                else:
                    while True:
                        try:
                            output_dict = openai.ChatCompletion.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system",
                                     "content": prompt_convincingness.replace("{{source_document}}",
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
        results = {}
        results[paper_id] = documents_annotated
        with open("gpt4_result_small/%s.json" % paper_id, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    random.seed(42)
    openai.api_key = "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"

    with open("../annotation_analysis/bryan_annotation_result.json") as f:
        bryan_results = json.load(f)
    with open("../annotation_analysis/zenan_annotation_result.json") as f:
        zenan_results = json.load(f)
    assert len(set(bryan_results.keys()).difference(set(zenan_results.keys()))) == 0
    samples_annotated_keys = bryan_results.keys()

    with open("../annotation_analysis/gpt4_annotation_data_small.json") as f:
        samples_all = json.load(f)
    # Evaluation data for agreement of GPT-4 with human annotators
    samples_gpt4 = {}
    for sample_key in samples_all.keys():
        if sample_key in samples_annotated_keys:
            samples_gpt4[sample_key] = samples_all[sample_key]
    annotating(samples_gpt4)

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
