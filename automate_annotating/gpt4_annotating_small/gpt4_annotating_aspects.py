import random
import openai
import time
import json
import os


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
        return "Overall"
    else:
        print("Cannot parse the facet correctly")
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
        print("Cannot parse the polarity correctly")
        return "Negative"


def parse_expressor(result):
    print(result)
    result = result.lower()
    if "self" in result:
        return "Self"
    elif "others" in result:
        return "Others"
    else:
        print("Cannot parse the expresser correctly")
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
        print("Cannot parse the convincingness correctly")
        return "Slightly Convincing"


def annotating_judgements(document, judgements):
    prompt_facet = open("../prompts/prompt_facet.txt").read()
    prompt_expresser = open("../prompts/prompt_expresser.txt").read()
    prompt_convincingness = open("../prompts/prompt_convincingness.txt").read()
    prompt_polarity = open("../prompts/prompt_polarity.txt").read()

    judgements_new = []
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
                                                                           document).replace(
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
                                                                              document).replace(
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
                        {"role": "system", "content": prompt_expresser.replace("{{source_document}}",
                                                                               document).replace(
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
                                                                      document).replace(
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
        judgements_new.append(judgement)
    return judgements_new


def annotating_all(samples, expressions):
    """
    :param samples: dict
    :return: none
    """
    print("Annotating count", len(samples), len(expressions))

    for paper_id, sample in samples.items():
        print(paper_id)

        annotated_documents = expressions[paper_id]
        documents_annotated = []
        for i, annotated_document in enumerate(annotated_documents):
            title = annotated_document["Document Title"]
            document = ""
            if i == 0 and title == sample["meta_review_title"]:
                document = sample["meta_review"]
            else:
                for review in sample["reviews"]:
                    if review["title"] == title:
                        document = review["comment"]
                        break
            judgements = annotated_document["Annotated Judgements"]
            if document == "":
                print("There is something wrong with the document title.")
            documents_annotated.append(
                {"Document Title": title, "Annotated Judgements": annotating_judgements(document, judgements)})

        results = {}
        results[paper_id] = documents_annotated
        with open("../gpt4_result_small/%s.json" % paper_id, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    random.seed(42)
    openai.api_key = "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"

    with open("../../annotation_data/annotation_data_small.json") as f:
        samples_all = json.load(f)

    f = open("experiment_ids_dev.txt")
    ids = f.read().split("\n")

    target_folder = "../gpt4_result_small"
    files_all = os.listdir(target_folder)
    annotated_expressions = {}
    for file in files_all:
        if file[:-5] in ids:
            annotated_expressions.update(json.load(open(os.path.join(target_folder, file))))

    # Evaluation data for agreement of GPT-4 with human annotators
    annotation_data = {}
    for sample_key in samples_all.keys():
        if sample_key in ids:
            annotation_data[sample_key] = samples_all[sample_key]
    annotating_all(annotation_data, annotated_expressions)
