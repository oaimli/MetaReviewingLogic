import sys

sys.path.append("../")

from annotation_analysis.annotator_behaviour import *

if __name__ == "__main__":
    with open("../annotation_analysis/bryan_annotation_result.json") as f:
        bryan_results = json.load(f)
    with open("../annotation_analysis/zenan_annotation_result.json") as f:
        zenan_results = json.load(f)
    with open("gpt4_result_small.json") as f:
        gpt4_results = json.load(f)

    bryan_results_share = {}
    zenan_results_share = {}
    gpt4_results_share = {}
    for key in list(
            set(bryan_results.keys()).intersection(set(zenan_results.keys())).intersection(set(gpt4_results.keys()))):
        bryan_results_share[key] = bryan_results[key]
        zenan_results_share[key] = zenan_results[key]
        gpt4_results_share[key] = gpt4_results[key]

    print("################ Annotator Bryan: ################")
    single_behaviour(bryan_results_share)

    print("################ Annotator Zenan: ################")
    single_behaviour(zenan_results_share)

    print("################ Annotator GPT-4: ################")
    single_behaviour(gpt4_results_share)

    print("################ Annotator Agreement Bryan and Zenan: ################")
    annotator_agreement(bryan_results_share, zenan_results_share)

    print("################ Annotator Agreement Bryan and GPT-4: ################")
    annotator_agreement(bryan_results_share, gpt4_results_share)

    print("################ Annotator Agreement Zenan and GPT-4: ################")
    annotator_agreement(zenan_results_share, gpt4_results_share)
