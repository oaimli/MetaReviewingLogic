import sys

sys.path.append("../../")

from annotation_analysis.annotator_behaviour import *

if __name__ == "__main__":
    for type in ["meta-review", "official-reviews", "others", "all"]:
        print("############", type, "#############")
        with open("../../annotation_analysis/bryan_annotation_result.json") as f:
            bryan_results = json.load(f)
        with open("../../annotation_analysis/zenan_annotation_result.json") as f:
            zenan_results = json.load(f)
        with open("prompting_strategy_01/gpt4_result_small.json") as f:
            gpt4_results = json.load(f)
        with open("../../annotation_data/annotation_data_small.json") as f:
            annotation_data = json.load(f)

        bryan_results_share = {}
        zenan_results_share = {}
        gpt4_results_share = {}
        annotation_data_share = {}
        shared_ids = list(
                set(bryan_results.keys()).intersection(set(zenan_results.keys())).intersection(set(gpt4_results.keys())))
        print("The count of shared ids", len(shared_ids))
        for key in shared_ids:
            bryan_results_share[key] = bryan_results[key]
            zenan_results_share[key] = zenan_results[key]
            gpt4_results_share[key] = gpt4_results[key]
            annotation_data_share[key] = annotation_data[key]

        for key in shared_ids:
            bryan_result = bryan_results_share[key]
            zenan_result = zenan_results_share[key]
            gpt4_result = gpt4_results_share[key]
            source_data = annotation_data_share[key]

            target_titles = []
            source_data_new = []
            if type == "meta-review":
                target_titles.append(source_data["meta_review_title"])
                source_data_new.append({"title": source_data["meta_review_title"], "content": source_data["meta_review"]})
            elif type == "official-reviews":
                for review in source_data["reviews"]:
                    if review["writer"] == "official_reviewer":
                        target_titles.append(review["title"])
                        source_data_new.append(
                            {"title": review["title"], "content": review["comment"]})
            elif type == "others":
                for review in source_data["reviews"]:
                    if review["writer"] != "official_reviewer":
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

            gpt4_result_new = []
            for result in gpt4_result:
                title = result["Document Title"]
                if title in target_titles:
                    gpt4_result_new.append(result)
            gpt4_results_share[key] = gpt4_result_new

        print("################ Annotator Bryan: ################")
        single_behaviour(bryan_results_share)

        print("################ Annotator Zenan: ################")
        single_behaviour(zenan_results_share)

        print("################ Annotator GPT-4: ################")
        single_behaviour(gpt4_results_share)

        print("################ Annotator Agreement Bryan and Zenan: ################")
        annotator_agreement(bryan_results_share, zenan_results_share, annotation_data_share)

        print("################ Annotator Agreement Bryan and GPT-4: ################")
        annotator_agreement(bryan_results_share, gpt4_results_share, annotation_data_share)

        print("################ Annotator Agreement Zenan and GPT-4: ################")
        annotator_agreement(zenan_results_share, gpt4_results_share, annotation_data_share)
