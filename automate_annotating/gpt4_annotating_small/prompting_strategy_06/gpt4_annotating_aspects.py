import random
import openai
import time
import json
import jsonlines


def parse_result(output):
    print("Output")
    print(output)
    with open("tmp.jsonl", "w") as f:
        f.write(output.strip())
    tmp = []
    try:
        with jsonlines.open("tmp.jsonl") as reader:
            for line in reader:
                tmp.append(line)
    except jsonlines.InvalidLineError as err:
        print("Jsonlines parsing error,", err)
    results = []
    for result in tmp:
        tmp = {}
        tmp["Content Expression"] = result["Content Expression"]
        tmp["Sentiment Expression"] = result["Sentiment Expression"]

        if "Criteria Facet" in result.keys():
            criteria_facet = result["Criteria Facet"].lower()

            if "novelty" in criteria_facet:
                criteria_facet = "Novelty"
            elif "soundness" in criteria_facet:
                criteria_facet = "Soundness"
            elif "clarity" in criteria_facet:
                criteria_facet = "Clarity"
            elif "advancement" in criteria_facet:
                criteria_facet = "Advancement"
            elif "compliance" in criteria_facet:
                criteria_facet = "Compliance"
            elif "overall" in criteria_facet:
                criteria_facet = "Overall"
            else:
                print("Cannot parse the facet correctly")
                criteria_facet = "Overall"
            tmp["Criteria Facet"] = criteria_facet
        else:
            tmp["Criteria Facet"] = "Overall"

        if "Sentiment Polarity" in result.keys():
            sentiment_polarity = result["Sentiment Polarity"].lower()
            if "strong" in sentiment_polarity and "negative" in sentiment_polarity:
                sentiment_polarity = "Strong negative"
            elif "strong" not in sentiment_polarity and "negative" in sentiment_polarity:
                sentiment_polarity = "Negative"
            elif "strong" in sentiment_polarity and "positive" in sentiment_polarity:
                sentiment_polarity = "Strong positive"
            elif "strong" not in sentiment_polarity and "positive" in sentiment_polarity:
                sentiment_polarity = "Positive"
            else:
                print("Cannot parse the polarity correctly")
                sentiment_polarity = "Negative"
            tmp["Sentiment Polarity"] = sentiment_polarity
        else:
            tmp["Sentiment Polarity"] = "Negative"

        if "Sentiment Expresser" in result.keys():
            sentiment_expresser = result["Sentiment Expresser"].lower()
            if "self" in sentiment_expresser:
                sentiment_expresser = "Self"
            elif "others" in sentiment_expresser:
                sentiment_expresser = "Others"
            else:
                print("Cannot parse the expresser correctly")
                sentiment_expresser = "Self"
            tmp["Sentiment Expresser"] = sentiment_expresser
        else:
            tmp["Sentiment Expresser"] = "Self"

        if "Convincingness" in result.keys():
            convincingness = result["Convincingness"].lower()
            if "not" in convincingness and "applicable" in convincingness:
                convincingness = "Not applicable"
            elif "not" in convincingness and "at all" in convincingness:
                convincingness = "Not at all"
            elif "slightly" in convincingness and "convincing" in convincingness:
                convincingness = "Slightly Convincing"
            elif "highly" in convincingness and "convincing" in convincingness:
                convincingness = "Highly Convincing"
            else:
                print("Cannot parse the convincingness correctly")
                convincingness = "Slightly Convincing"
            tmp["Convincingness"] = convincingness
        else:
            tmp["Convincingness"] = "Slightly Convincing"

        results.append(tmp)
    return results


def annotating_judgements(document, judgements):
    prompt_aspects = open("prompts/prompt_aspects.txt").read()

    judgement_expressions = ""
    for i, judgement in enumerate(judgements):
        judgement["Criteria Facet"] = ""
        judgement["Sentiment Polarity"] = ""
        judgement["Sentiment Expresser"] = ""
        judgement["Convincingness"] = ""
        if i == len(judgements) - 1:
            tmp = json.dumps(judgement)
            judgement_expressions += tmp
        else:
            tmp = json.dumps(judgement) + "\n"
            judgement_expressions += tmp
    print("GPT-4/Human annotated expressions")
    print(judgement_expressions)

        # get criteria facet
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt_aspects.replace("{{source_document}}",
                                                                       document).replace(
                        "{{judgement_expressions}}", judgement_expressions)}
                ],
                n = 3
            )
            results = []
            for output in output_dict['choices']:
                judgements_new = parse_result(output['message']['content'])
                if len(judgements) == len(judgements_new):
                    results = judgements_new
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    print(len(judgements), len(results))
    return results


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
        with open("result/%s.json" % paper_id, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    random.seed(42)
    openai.api_key = "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"

    with open("../../../annotation_data/annotation_data_small.json") as f:
        samples_all = json.load(f)

    f = open("experiment_ids_dev.txt")
    ids = f.read().split("\n")

    with open("../../../annotation_analysis/bryan_annotation_result.json") as f:
        samples_annotated = json.load(f)
    annotated_expressions = {}
    for id, sample in samples_annotated.items():
        if id in ids:
            annotated_expressions.update({id: sample})

    # Evaluation data for agreement of GPT-4 with human annotators
    annotation_data = {}
    for sample_key in samples_all.keys():
        if sample_key in ids:
            annotation_data[sample_key] = samples_all[sample_key]
    annotating_all(annotation_data, annotated_expressions)
