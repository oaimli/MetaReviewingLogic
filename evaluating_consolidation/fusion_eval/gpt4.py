import json
import openai

def annotating_with_judgements(source_judgements, judgement):
    sentiment_level = ""
    return sentiment_level

def annotating_with_source_text(source_text, judgement):
    sentiment_level = ""
    return sentiment_level



if __name__ == "__main__":
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
