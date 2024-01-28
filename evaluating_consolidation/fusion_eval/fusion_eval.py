import json
import openai
import time


def parse_result(output):
    with open("fusion_eval_tmp.json", "w") as f:
        f.write(output.strip())
    result = {}
    try:
        with open("fusion_eval_tmp.json") as f:
            result = json.load(f)
        return result
    except:
        return result


def annotating_with_judgements(source_judgements, judgement, prompt_file="prompt_for_judgements.txt"):
    prompt = open(prompt_file).read()
    content_expression = judgement["Content Expression"]
    sentiment_level = judgement["Sentiment Polarity"]
    tmp = []
    for item in source_judgements:
        tmp.append(json.dumps(item))
    tmp = "\n".join(tmp)
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt.replace("{{source_judgements}}",
                                                                       tmp).replace(
                        "{{content_expression}}", content_expression)}
                ],
                n = 3
            )
            result = {}
            for output in output_dict['choices']:
                tmp = parse_result(output['message']['content'])
                if len(tmp) > 0:
                    result = tmp
                    break
            if "Sentiment Level" in result.keys() and "Content Expression" in result.keys():
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    if result["Sentiment Level"] == sentiment_level:
        return 1
    else:
        return 0


def annotating_with_source_text(source_text, judgement, prompt_file="prompt_for_source_texts.txt"):
    prompt = open(prompt_file).read()
    content_expression = judgement["Content Expression"]
    sentiment_level = judgement["Sentiment Polarity"]
    while True:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt.replace("{{source_texts}}",
                                                                 source_text).replace(
                        "{{content_expression}}", content_expression)}
                ],
                n=3
            )
            result = {}
            for output in output_dict['choices']:
                tmp = parse_result(output['message']['content'])
                if len(tmp) > 0:
                    result = tmp
                    break
            if "Sentiment Level" in result.keys() and "Content Expression" in result.keys():
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    if result["Sentiment Level"] == sentiment_level:
        return 1
    else:
        return 0



if __name__ == "__main__":
    # This is for sentiment fusion analysis
    openai.api_key = "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"

    with open("../../annotation_analysis/bryan_annotation_result.json") as f:
        bryan_results = json.load(f)
    with open("../../annotation_analysis/zenan_annotation_result.json") as f:
        zenan_results = json.load(f)
    with open("../../annotation_data/annotation_data_small.json") as f:
        annotation_data_tmp = json.load(f)
    annotation_data = {}
    for key in annotation_data_tmp.keys():
        if key in bryan_results.keys() and key in zenan_results.keys():
            annotation_data[key] = annotation_data_tmp[key]
    print("Bryan", len(bryan_results), "Zenan", len(zenan_results), "Annotation data", len(annotation_data))

    judgements_bryan = []
    judgements_zenan = []
    for key in zenan_results.keys():
        bryan_result = bryan_results[key]
        meta_review_judgements_bryan = bryan_result[0]["Annotated Judgements"]
        source_judgements_bryan = []
        for judgement in bryan_result[1:]:
            source_judgements_bryan.extend(judgement["Annotated Judgements"])

        zenan_result = zenan_results[key]
        meta_review_judgements_zenan = zenan_result[0]["Annotated Judgements"]
        source_judgements_zenan = []
        for judgement in zenan_result[1:]:
            source_judgements_zenan.extend(judgement["Annotated Judgements"])

        annotation_sample = annotation_data[key]
        source_texts = []
        for review in annotation_sample["reviews"]:
            if review["writer"] == "official_reviewer":
                source_texts.append(review["comment"])
        source_texts = "\n".join(source_texts)

        for meta_review_judgement_bryan in meta_review_judgements_bryan:
            source_judgements = []
            for judgement in source_judgements_bryan:
                if judgement["Criteria Facet"] == meta_review_judgement_bryan["Criteria Facet"]:
                    source_judgements.append(judgement)
            instance = {}
            instance["meta_review_judgement"] = meta_review_judgement_bryan
            instance["source_judgements"] = source_judgements
            instance["source_texts"] = source_texts
            judgements_bryan.append(instance)

        for meta_review_judgement_zenan in meta_review_judgements_zenan:
            source_judgements = []
            for judgement in source_judgements_zenan:
                if judgement["Criteria Facet"] == meta_review_judgement_zenan["Criteria Facet"]:
                    source_judgements.append(judgement)
            instance = {}
            instance["meta_review_judgement"] = meta_review_judgement_zenan
            instance["source_judgements"] = source_judgements
            instance["source_texts"] = source_texts
            judgements_zenan.append(instance)

    print(len(judgements_bryan), len(judgements_zenan))

    facets = ["Advancement", "Soundness", "Novelty", "Overall", "Clarity", "Compliance"]
    # facets = ["Advancement"]
    for facet in facets:
        print(facet)

        # instances_bryan_facet = []
        # for instance in judgements_bryan:
        #     judgement = instance["meta_review_judgement"]
        #     if judgement["Criteria Facet"] == facet:
        #         instances_bryan_facet.append(instance)
        # print("Instances in this facet", len(instances_bryan_facet))
        #
        # print("Bryan with source judgements")
        # correct = 0
        # for instance in instances_bryan_facet:
        #     correct += annotating_with_judgements(instance["source_judgements"], instance["meta_review_judgement"])
        # print("Correct", correct, correct / len(instances_bryan_facet))
        #
        # print("Bryan with source texts")
        # correct = 0
        # for instance in instances_bryan_facet:
        #     correct += annotating_with_source_text(instance["source_texts"], instance["meta_review_judgement"])
        # print("Correct", correct, correct / len(instances_bryan_facet))

        instances_zenan_facet = []
        for instance in judgements_zenan:
            judgement = instance["meta_review_judgement"]
            if judgement["Criteria Facet"] == facet:
                instances_zenan_facet.append(instance)
        print("Judgements in this facet", len(instances_zenan_facet))

        print("Zenan with source judgements")
        correct = 0
        for instance in instances_zenan_facet:
            correct += annotating_with_judgements(instance["source_judgements"], instance["meta_review_judgement"])

        print("Correct", correct, correct / len(instances_zenan_facet))

        print("Zenan with source texts")
        correct = 0
        for instance in instances_zenan_facet:
            correct += annotating_with_source_text(instance["source_texts"], instance["meta_review_judgement"])
        print("Correct", correct, correct / len(instances_zenan_facet))

