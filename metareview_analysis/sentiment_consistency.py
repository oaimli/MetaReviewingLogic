import json
import numpy as np
import jsonlines
from scipy import spatial

with open("../annotation_analysis/bryan_annotation_result.json") as f:
    bryan_results = json.load(f)
with open("../annotation_analysis/zenan_annotation_result.json") as f:
    zenan_results = json.load(f)
# with open("prompting_strategy_01/gpt4_result_small.json") as f:
#     gpt4_results = json.load(f)
with open("../annotation_data/annotation_data_small.json") as f:
    annotation_data = json.load(f)

bryan_results_share = {}
zenan_results_share = {}
# gpt4_results_share = {}
annotation_data_share = {}
acceptances = {}

# shared_ids = list(
#     set(bryan_results.keys()).intersection(set(zenan_results.keys())).intersection(set(gpt4_results.keys())))
shared_ids = list(
    set(bryan_results.keys()).intersection(set(zenan_results.keys())))
for key in shared_ids:
    bryan_results_share[key] = bryan_results[key]
    zenan_results_share[key] = zenan_results[key]
    # gpt4_results_share[key] = gpt4_results[key]
    annotation_data_share[key] = annotation_data[key]

# print("Bryan", len(bryan_results_share), "Zenan", len(zenan_results_share), "GPT-4", len(gpt4_results_share),
#       "Annotation data", len(annotation_data_share))
print("Bryan", len(bryan_results_share), "Zenan", len(zenan_results_share), "Annotation data", len(annotation_data_share))

type = "official-reviews"
for key in shared_ids:
    bryan_result = bryan_results_share[key]
    zenan_result = zenan_results_share[key]
    # gpt4_result = gpt4_results_share[key]
    source_data = annotation_data_share[key]

    acceptances[key] = source_data["paper_acceptance"]

    target_titles = []
    source_data_new = []
    if type == "meta-review":
        target_titles.append(source_data["meta_review_title"])
        source_data_new.append(
            {"title": source_data["meta_review_title"], "content": source_data["meta_review"]})
    elif type == "official-reviews":
        for review in source_data["reviews"]:
            if review["writer"] == "official_reviewer":
                target_titles.append(review["title"])
                source_data_new.append(
                    {"title": review["title"], "content": review["comment"]})
    elif type == "discussions":
        for review in source_data["reviews"]:
            if review["writer"] != "official_reviewer":
                target_titles.append(review["title"])
                source_data_new.append(
                    {"title": review["title"], "content": review["comment"]})
    elif type == "others":
        for review in source_data["reviews"]:
            target_titles.append(review["title"])
            source_data_new.append(
                {"title": review["title"], "content": review["comment"]})
    else:
        target_titles.append(source_data["meta_review_title"])
        source_data_new.append(
            {"title": source_data["meta_review_title"], "content": source_data["meta_review"]})
        for review in source_data["reviews"]:
            target_titles.append(review["title"])
            source_data_new.append(
                {"title": review["title"], "content": review["comment"]})

    annotation_data_share[key] = source_data_new

    bryan_result_new = []
    for result in bryan_result:
        title = result["Document Title"]
        if title in target_titles:
            bryan_result_new.append(result)
    bryan_results_share[key] = bryan_result_new

    zenan_result_new = []
    for result in zenan_result:
        title = result["Document Title"]
        if title in target_titles:
            zenan_result_new.append(result)
    zenan_results_share[key] = zenan_result_new

    # gpt4_result_new = []
    # for result in gpt4_result:
    #     title = result["Document Title"]
    #     if title in target_titles:
    #         gpt4_result_new.append(result)
    # gpt4_results_share[key] = gpt4_result_new

conflicts = {}
with jsonlines.open("../../HumanAnnotation/mrg_judgement/annotation/sampled_data.jsonl") as reader:
    for line in reader:
        conflicts[line["paper_id"][10:]] = line["contradict"]


def get_score(result):
    scores = {"Advancement": [], "Soundness": [], "Novelty": [], "Overall": [], "Clarity": [], "Compliance": []}
    result_len = len(result) # the number of annotated documents for one sample
    for i in range(0, result_len):
        result_document_i = result[i]
        facet_in_document_i = {"Advancement": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0}, "Soundness": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0}, "Novelty": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0}, "Overall": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0}, "Clarity": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0},
                             "Compliance": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0}}
        print("i", len(result_document_i["Annotated Judgements"]))
        print(result_document_i["Annotated Judgements"])
        for judgement_i in result_document_i["Annotated Judgements"]:
            facet_in_document_i[judgement_i["Criteria Facet"]][judgement_i["Sentiment Polarity"]] = facet_in_document_i[judgement_i["Criteria Facet"]].get(judgement_i["Sentiment Polarity"]) + 1
        for j in range(i+1, result_len):
            result_document_j = result[j]
            facet_in_document_j = {
                "Advancement": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0},
                "Soundness": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0},
                "Novelty": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0},
                "Overall": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0},
                "Clarity": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0},
                "Compliance": {"Positive": 0, "Negative": 0, "Strong positive": 0, "Strong negative": 0}}
            print("j", len(result_document_j["Annotated Judgements"]))
            print(result_document_j["Annotated Judgements"])
            for judgement in result_document_j["Annotated Judgements"]:
                facet_in_document_j[judgement["Criteria Facet"]][judgement["Sentiment Polarity"]] = facet_in_document_j[
                                                                                                        judgement[
                                                                                                            "Criteria Facet"]].get(
                    judgement["Sentiment Polarity"]) + 1
            print(facet_in_document_i)
            print(facet_in_document_j)
            for key in facet_in_document_i:
                print(key)
                v_i = list(facet_in_document_i[key].values())
                v_j = list(facet_in_document_j[key].values())
                co = 1 - spatial.distance.cosine(v_i, v_j)
                tmp = scores[key]
                tmp.append(co)
                scores[key] = tmp

    return scores

mean_with_conflicts_bryan = {}
variance_with_conflicts_bryan = {}
mean_without_conflicts_bryan = {}
variance_without_conflicts_bryan = {}
mean_with_conflicts_zenan = {}
variance_with_conflicts_zenan = {}
mean_without_conflicts_zenan = {}
variance_without_conflicts_zenan = {}
for key in shared_ids:
    bryan_result = bryan_results_share[key]
    zenan_result = zenan_results_share[key]

    print(key)
    bryan_corrs = get_score(bryan_result)
    print(bryan_corrs)
    zenan_corrs = get_score(zenan_result)

    for facet_key in bryan_corrs:
        bryan_corr = bryan_corrs[facet_key]

        mean = np.mean(bryan_corr)
        var = np.var(bryan_corr)
        if conflicts[key] == 1:
            tmp = mean_with_conflicts_bryan.get(facet_key, [])
            tmp.append(mean)
            mean_with_conflicts_bryan[facet_key] = tmp
            tmp = variance_with_conflicts_bryan.get(facet_key, [])
            tmp.append(var)
            variance_with_conflicts_bryan[facet_key] = tmp
        else:
            tmp = mean_without_conflicts_bryan.get(facet_key, [])
            tmp.append(mean)
            mean_without_conflicts_bryan[facet_key] = tmp
            tmp = variance_without_conflicts_bryan.get(facet_key, [])
            tmp.append(var)
            variance_without_conflicts_bryan[facet_key] = tmp

        zenan_corr = zenan_corrs[facet_key]
        mean = np.mean(zenan_corr)
        var = np.var(zenan_corr)
        if conflicts[key] == 1:
            tmp = mean_with_conflicts_zenan.get(facet_key, [])
            tmp.append(mean)
            mean_with_conflicts_zenan[facet_key] = tmp
            tmp = variance_with_conflicts_zenan.get(facet_key, [])
            tmp.append(var)
            variance_with_conflicts_zenan[facet_key] = tmp
        else:
            tmp = mean_without_conflicts_zenan.get(facet_key, [])
            tmp.append(mean)
            mean_without_conflicts_zenan[facet_key] = tmp
            tmp = variance_without_conflicts_zenan.get(facet_key, [])
            tmp.append(var)
            variance_without_conflicts_zenan[facet_key] = tmp

for facet_key in mean_with_conflicts_bryan:
    print(facet_key)
    # mean_with_conflicts_bryan = {}
    # print(mean_with_conflicts_bryan[facet_key])
    print(np.mean(mean_with_conflicts_bryan[facet_key]))
    # variance_with_conflicts_bryan = {}
    # print(variance_with_conflicts_bryan[facet_key])
    print(np.mean(variance_with_conflicts_bryan[facet_key]))
    # mean_without_conflicts_bryan = {}
    # print(mean_without_conflicts_bryan[facet_key])
    print(np.mean(mean_without_conflicts_bryan[facet_key]))
    # variance_without_conflicts_bryan = {}
    # print(variance_without_conflicts_bryan[facet_key])
    print(np.mean(variance_without_conflicts_bryan[facet_key]))

    # mean_with_conflicts_zenan = {}
    print(np.mean(mean_with_conflicts_zenan[facet_key]))
    # variance_with_conflicts_zenan = {}
    print(np.mean(variance_with_conflicts_zenan[facet_key]))
    # mean_without_conflicts_zenan = {}
    print(np.mean(mean_without_conflicts_zenan[facet_key]))
    # variance_without_conflicts_zenan = {}
    print(np.mean(variance_without_conflicts_zenan[facet_key]))