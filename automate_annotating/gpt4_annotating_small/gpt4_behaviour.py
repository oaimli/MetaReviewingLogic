import sys

sys.path.append("../../")

from annotation_analysis.annotator_behaviour import *

if __name__ == "__main__":
    for type in ["meta-review", "official-reviews", "others", "all"]:
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
        for key in shared_ids:
            bryan_results_share[key] = bryan_results[key]
            zenan_results_share[key] = zenan_results[key]
            gpt4_results_share[key] = gpt4_results[key]
            annotation_data_share[key] = annotation_data[key]

        print("Bryan", len(bryan_results_share), "Zenan", len(zenan_results_share), "GPT-4", len(gpt4_results_share),
              "Annotation data", len(annotation_data_share))

        print("############", type, "#############")
        for key in shared_ids:
            bryan_result = bryan_results_share[key]
            zenan_result = zenan_results_share[key]
            gpt4_result = gpt4_results_share[key]
            source_data = annotation_data_share[key]

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
        result_bryan = single_behaviour(bryan_results_share)

        print("################ Annotator Zenan: ################")
        result_zenan = single_behaviour(zenan_results_share)

        print("################ Annotator GPT-4: ################")
        result_gpt4 = single_behaviour(gpt4_results_share)

        print("################ Annotator Agreement Bryan and Zenan: ################")
        result_bz = annotator_agreement(bryan_results_share, zenan_results_share, annotation_data_share, type)

        print("################ Annotator Agreement Bryan and GPT-4: ################")
        result_bg = annotator_agreement(bryan_results_share, gpt4_results_share, annotation_data_share, type)

        print("################ Annotator Agreement Zenan and GPT-4: ################")
        result_zg = annotator_agreement(zenan_results_share, gpt4_results_share, annotation_data_share, type)

        print("################ %s, Overall results for single behaviour, HumanAnnotator1, HumanAnnotator2 and GPT-4: ################" % type)
        for key in result_bryan:
            print(key, "------", result_bryan[key], result_zenan[key], result_gpt4[key])

        print(
            "################ %s, Overall results for agreement, A1<->A2, A1<->GPT-4 and A2<->GPT-4: ################" % type)
        for key in result_bz:
            print(key, "------", result_bz[key], result_bg[key], result_zg[key])

