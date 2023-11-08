import json
import numpy as np
import pandas as pd


def avg_annotated_documents(results):
    count_documents = 0
    count_judgements = 0
    for k, v in results.items():
        count_documents += len(v)
        for document in v:
            count_judgements += len(document["Annotated Judgements"])
    return count_documents / len(results.keys()), count_judgements / count_documents, count_judgements


def avg_lens(results):
    count_judgement = 0
    lens_content = []
    lens_sentiment = []
    for k, v in results.items():
        for document in v:
            # print(document["Document Title"])
            for judgement in document["Annotated Judgements"]:
                count_judgement += 1
                lens_content.append(len(judgement["Content Expression"].split()))
                lens_sentiment.append(len(judgement["Sentiment Expression"].split()))
    return np.mean(lens_content), np.mean(lens_sentiment)


def distribution(results):
    criteria_facets = []
    sentiment_polarities = []
    sentiment_expressers = []
    convincingnesses = []

    for k, v in results.items():
        for document in v:
            # print(document["Document Title"])
            for judgement in document["Annotated Judgements"]:
                criteria_facets.append(judgement["Criteria Facet"])
                sentiment_expressers.append(judgement["Sentiment Expresser"])
                sentiment_polarities.append(judgement["Sentiment Polarity"])
                convincingnesses.append(judgement["Convincingness"])

    return pd.value_counts(criteria_facets), pd.value_counts(sentiment_expressers), pd.value_counts(
        sentiment_polarities), pd.value_counts(convincingnesses)


if __name__ == "__main__":
    with open("bryan_annotation_result.json") as f:
        bryan_results = json.load(f)
    print("Annotator Bryan:")
    print("Average annotated documents per sample: ", avg_annotated_documents(bryan_results)[0])
    print("All annotated judgements: ", avg_annotated_documents(bryan_results)[2])
    print("Average annotated judgements per document: ", avg_annotated_documents(bryan_results)[1])
    print("Average lens of content and sentiment: ", avg_lens(bryan_results))
    print("Distribution of criteria facets: \n", distribution(bryan_results)[0])
    print("Distribution of sentiment expressers: \n", distribution(bryan_results)[1])
    print("Distribution of sentiment polarities: \n", distribution(bryan_results)[2])
    print("Distribution of convincingnesses: \n", distribution(bryan_results)[3])

    with open("zenan_annotation_result.json") as f:
        zenan_results = json.load(f)
    print("Annotator Zenan:")
    print("Average annotated documents per sample: ", avg_annotated_documents(zenan_results)[0])
    print("All annotated judgements: ", avg_annotated_documents(zenan_results)[2])
    print("Average annotated judgements per document: ", avg_annotated_documents(zenan_results)[1])
    print("Average lens of content and sentiment: ", avg_lens(zenan_results))
    print("Distribution of criteria facets: \n", distribution(zenan_results)[0])
    print("Distribution of sentiment expressers: \n", distribution(zenan_results)[1])
    print("Distribution of sentiment polarities: \n", distribution(zenan_results)[2])
    print("Distribution of convincingnesses: \n", distribution(zenan_results)[3])
