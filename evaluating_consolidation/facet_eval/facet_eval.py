import json
import time
import jsonlines
import openai
from tqdm import tqdm
from scipy import spatial

def facet_score(judgements_of_reference, judgements_of_candidate):
    representation_reference = {"Novelty Strong negative": 0, "Novelty Negative": 0, "Novelty Positive": 0, "Novelty Strong positive": 0, "Novelty Not mentioned": 1,
                                "Soundness Strong negative": 0, "Soundness Negative": 0, "Soundness Positive": 0, "Soundness Strong positive": 0, "Soundness Not mentioned": 1,
                                "Clarity Strong negative": 0, "Clarity Negative": 0, "Clarity Positive": 0, "Clarity Strong positive": 0, "Clarity Not mentioned": 1,
                                "Advancement Strong negative": 0, "Advancement Negative": 0, "Advancement Positive": 0, "Advancement Strong positive": 0, "Advancement Not mentioned": 1,
                                "Compliance Strong negative": 0, "Compliance Negative": 0, "Compliance Positive": 0, "Compliance Strong positive": 0, "Compliance Not mentioned": 1,
                                "Overall Strong negative": 0, "Overall Negative": 0, "Overall Positive": 0, "Overall Strong positive": 0, "Overall Not mentioned": 1}
    for judgement in judgements_of_reference:
        tmp = judgement["Criteria Facet"] + " " + judgement["Sentiment Polarity"]
        not_mention = judgement["Criteria Facet"] + " Not mentioned"
        representation_reference[tmp] = representation_reference[tmp] + 1
        representation_reference[not_mention] = 0

    representation_candidate = {"Novelty Strong negative": 0, "Novelty Negative": 0, "Novelty Positive": 0, "Novelty Strong positive": 0, "Novelty Not mentioned": 1,
                                "Soundness Strong negative": 0, "Soundness Negative": 0, "Soundness Positive": 0, "Soundness Strong positive": 0, "Soundness Not mentioned": 1,
                                "Clarity Strong negative": 0, "Clarity Negative": 0, "Clarity Positive": 0, "Clarity Strong positive": 0, "Clarity Not mentioned": 1,
                                "Advancement Strong negative": 0, "Advancement Negative": 0, "Advancement Positive": 0, "Advancement Strong positive": 0, "Advancement Not mentioned": 1,
                                "Compliance Strong negative": 0, "Compliance Negative": 0, "Compliance Positive": 0, "Compliance Strong positive": 0, "Compliance Not mentioned": 1,
                                "Overall Strong negative": 0, "Overall Negative": 0, "Overall Positive": 0, "Overall Strong positive": 0, "Overall Not mentioned": 1}
    for judgement in judgements_of_candidate:
        tmp = judgement["Criteria Facet"] + " " + judgement["Sentiment Polarity"]
        not_mention = judgement["Criteria Facet"] + " Not mentioned"
        representation_candidate[tmp] = representation_candidate[tmp] + 1
        representation_candidate[not_mention] = 0
    # print(list(representation_candidate.values()))
    # print(list(representation_reference.values()))
    similarity = 1 - spatial.distance.cosine(list(representation_candidate.values()), list(representation_reference.values()))
    return {"facet_eval": similarity}

def annotating_expressions(meta_review):
    prompt_format = open("prompt_expression.txt").read()
    i = 0
    while True:
        print("try", i)
        i += 1
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt_format.replace("{{input_document}}", meta_review)}
                ],
                n=5
            )
            results = []
            for output in output_dict['choices']:
                output_content = output['message']['content'].replace("\n\n", "\n")
                # print(output_content)
                with open("tmp.jsonl", "w") as f:
                    f.write(output_content.strip())
                tmp = []
                try:
                    with jsonlines.open("tmp.jsonl") as reader:
                        for line in reader:
                            tmp.append(line)
                except jsonlines.InvalidLineError as err:
                    print("Jsonlines parsing error,", err)
                if len(tmp) > len(results):
                    results = tmp
            if len(results) > 0:
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(len(results))
    judgements = []
    for line in results:
        # print(line)
        if "content_expression" in line.keys() and "sentiment_expression" in line.keys():
            if line["content_expression"].strip() != "" and line["sentiment_expression"].strip() != "":
                judgements.append(
                    {"Content Expression": line["content_expression"],
                     "Sentiment Expression": line["sentiment_expression"],
                     "Criteria Facet": "",
                     "Sentiment Polarity": "", "Sentiment Expresser": "", "Convincingness": ""})
    return judgements


def parse_result(output):
    # print("Output")
    # print(output)
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

def annotating_facets(meta_review, judgements):
    prompt_aspects = open("prompt_aspects.txt").read()

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
    # print("GPT-4/Human annotated expressions")
    # print(judgement_expressions)

    # get criteria facet
    i = 0
    while True:
        print("try", i)
        i += 1
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt_aspects.replace("{{input_document}}",
                                                                       meta_review).replace(
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
            if len(results) > 0:
                break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    # print(len(judgements), len(results))
    return results


if __name__ == "__main__":
    openai.api_key = "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"

    # with open("../../enhancing_prompting/test_data.json") as f:
    #     test_samples = json.load(f)
    # for key in tqdm(test_samples, desc="test samples"):
    #     expressions = annotating_expressions(test_samples[key]["meta_review"])
    #     judgements = annotating_facets(test_samples[key]["meta_review"], expressions)
    #     test_samples[key]["gpt4_judgements"] = judgements
    # with open("../../enhancing_prompting/test_data.json", "w") as f:
    #     json.dump(test_samples, f, indent=4)


    with open("../../enhancing_prompting/results/generation_gpt35_prompt_naive.json") as f:
        generations_prompt_naive = json.load(f)
    for key in tqdm(generations_prompt_naive, desc="prompt naive"):
        expressions = annotating_expressions(generations_prompt_naive[key]["generation"])
        judgements = annotating_facets(generations_prompt_naive[key]["generation"], expressions)
        result = {}
        result[key] = judgements
        with open("../judgements/generation_gpt35_prompt_naive/%s.json" % key, "w") as f:
            json.dump(result, f, indent=4)


    # with open("../../enhancing_prompting/results/generation_gpt35_prompt_llm.json") as f:
    #     generations_prompt_llm = json.load(f)
    #
    # for key in tqdm(generations_prompt_llm, desc="prompt llm"):
    #     print("expressions")
    #     expressions = annotating_expressions(generations_prompt_llm[key]["generation"])
    #     print("facets")
    #     judgements = annotating_facets(generations_prompt_llm[key]["generation"], expressions)
    #     result = {}
    #     result[key] = judgements
    #     with open("../judgements/generation_gpt35_prompt_llm/%s.json" % key, "w") as f:
    #         json.dump(result, f, indent=4)

