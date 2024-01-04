import json
import pandas as pd

with open(".../annotation_analysis/bryan_annotation_result.json") as f:
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

type = "meta-review"
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

acceptances_all = []
bryan_facets = []
zenan_facets = []
bryan_facet_in_documents = []
zenan_facet_in_documents = []
for key in shared_ids:
    acceptances_all.append(acceptances[key])
    bryan_result = bryan_results_share[key]
    zenan_result = zenan_results_share[key]

    for result_document in bryan_result:
        facet_in_document = {}
        for judgement in result_document["Annotated Judgements"]:
            bryan_facets.append(judgement["Criteria Facet"])
            facet_in_document[judgement["Criteria Facet"]] = 1
        bryan_facet_in_documents.append(facet_in_document)
    for result_document in zenan_result:
        facet_in_document = {}
        for judgement in result_document["Annotated Judgements"]:
            zenan_facets.append(judgement["Criteria Facet"])
            facet_in_document[judgement["Criteria Facet"]] = 1
        zenan_facet_in_documents.append(facet_in_document)

print(print(pd.value_counts(bryan_facets, normalize=True)))

