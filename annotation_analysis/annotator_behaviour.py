import json
import numpy as np
from rouge_score import rouge_scorer
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder


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

def single_behaviour(results):
    print("Average annotated documents per sample: ", avg_annotated_documents(results)[0])
    print("All annotated judgements: ", avg_annotated_documents(results)[2])
    print("Average annotated judgements per document: ", avg_annotated_documents(results)[1])
    print("Average lens of content and sentiment: ", avg_lens(results))
    print("Distribution of criteria facets: \n", distribution(results)[0])
    print("Distribution of sentiment expressers: \n", distribution(results)[1])
    print("Distribution of sentiment polarities: \n", distribution(results)[2])
    print("Distribution of convincingnesses: \n", distribution(results)[3])


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



with open("bryan_annotation_result.json") as f:
    bryan_results = json.load(f)

with open("zenan_annotation_result.json") as f:
    zenan_results = json.load(f)

scorer = rouge_scorer.RougeScorer(["rouge2", "rouge1", "rougeLsum"], use_stemmer=True)

sample_indexes = bryan_results.keys()
overlapped_document_count = []
judgements_bryan_share = []
judgements_zenan_share = []
for sample_index in sample_indexes:
    documents_bryan = bryan_results[sample_index]
    documents_zenan = zenan_results[sample_index]

    titles_bryan = set([])
    judgements_bryan = []
    for document in documents_bryan:
        titles_bryan.add(document["Document Title"])
        judgements_new = []
        for judgement in document["Annotated Judgements"]:
            judgement_new = {}
            for k, v in judgement.items():
                judgement_new[k] = v
            judgement_new["Document Title"] = document["Document Title"]
            judgement_new["id"] = sample_index
            judgements_new.append(judgement_new)
        judgements_bryan.extend(judgements_new)
    titles_zenan = set([])
    judgements_zenan = []
    for document in documents_zenan:
        titles_zenan.add(document["Document Title"])
        judgements_new = []
        for judgement in document["Annotated Judgements"]:
            judgement_new = {}
            for k, v in judgement.items():
                judgement_new[k] = v
            judgement_new["Document Title"] = document["Document Title"]
            judgement_new["id"] = sample_index
            judgements_new.append(judgement_new)
        judgements_zenan.extend(judgements_new)
    overlap = titles_bryan.intersection(titles_zenan)
    overlapped_document_count.append(len(overlap))

    for i, judgement_bryan in enumerate(judgements_bryan):
        rouges = []
        judgements_target = []
        for j, judgement_zenan in enumerate(judgements_zenan):
            if judgement_bryan["Document Title"] == judgement_zenan["Document Title"]:
                tmp_i = judgement_bryan["Content Expression"] + " " + judgement_bryan["Sentiment Expression"]
                tmp_j = judgement_zenan["Content Expression"] + " " + judgement_zenan["Sentiment Expression"]
                scores = scorer.score(tmp_j, tmp_i)
                rouges.append(scores["rouge2"].fmeasure + scores["rouge1"].fmeasure + scores["rougeLsum"].fmeasure)
                judgements_target.append(judgement_zenan)

        if len(rouges) > 0:
            judgement_target = judgements_target[rouges.index(max(rouges))]
            if max(rouges) > 1.0:
                print(judgement_bryan)
                print(judgement_target)
                print("\n")
                judgements_bryan_share.append(judgement_bryan)
                judgements_zenan_share.append(judgement_target)


criteria_facet_same = []
sentiment_level_same = []
sentiment_expresser_same = []
convincingness_same = []

sentiment_polarity_same = []

sentiment_levels_bryan = []
sentiment_levels_zenan = []
convincingness_levels_bryan = []
convincingness_levels_zenan = []
for judgement_bryan_share, judgement_zenan_share in zip(judgements_bryan_share, judgements_zenan_share):
    criteria_facet_bryan = judgement_bryan_share["Criteria Facet"]
    sentiment_expresser_bryan = judgement_bryan_share["Sentiment Expresser"]
    sentiment_polarity_bryan = judgement_bryan_share["Sentiment Polarity"]
    convincingness_bryan = judgement_bryan_share["Convincingness"]
    criteria_facet_zenan = judgement_zenan_share["Criteria Facet"]
    sentiment_expresser_zenan = judgement_zenan_share["Sentiment Expresser"]
    sentiment_polarity_zenan = judgement_zenan_share["Sentiment Polarity"]
    convincingness_zenan = judgement_zenan_share["Convincingness"]

    if criteria_facet_bryan == criteria_facet_zenan:
        criteria_facet_same.append(criteria_facet_bryan)

    if sentiment_polarity_bryan == sentiment_polarity_zenan:
        sentiment_level_same.append(sentiment_polarity_bryan)

    if sentiment_expresser_bryan == sentiment_expresser_zenan:
        sentiment_expresser_same.append(sentiment_expresser_zenan)

    if convincingness_bryan == convincingness_zenan:
        convincingness_same.append(convincingness_zenan)

    if "ositive" in sentiment_polarity_bryan and "ositive" in sentiment_polarity_zenan:
        sentiment_polarity_same.append("Positive")
    if "egative" in sentiment_polarity_bryan and "egative" in sentiment_polarity_zenan:
        sentiment_polarity_same.append("Negative")

    sentiment_levels_bryan.append(sentiment_polarity_bryan)
    sentiment_levels_zenan.append(sentiment_polarity_zenan)

    convincingness_levels_bryan.append(convincingness_bryan)
    convincingness_levels_zenan.append(convincingness_zenan)



print("Overlapped annotated documents in each sample: ", len(overlapped_document_count), np.mean(overlapped_document_count))
print("All shared judgements: ", len(judgements_bryan_share), len(judgements_bryan_share))
print("Shared criteria_facet in judgements: ", len(criteria_facet_same))
print(pd.value_counts(criteria_facet_same))
print("Shared sentiment_level in judgements: ", len(sentiment_level_same))
print(pd.value_counts(sentiment_level_same))
print("Shared sentiment_expresser in judgements: ", len(sentiment_expresser_same))
print(pd.value_counts(sentiment_expresser_same))
print("Shared convincingness in judgements: ", len(convincingness_same))
print(pd.value_counts(convincingness_same))
print("Shared sentiment_polarity in judgements: ", len(sentiment_polarity_same))
print(pd.value_counts(sentiment_polarity_same))

le_sentiment = LabelEncoder()
le_sentiment.fit(sentiment_levels_bryan + sentiment_levels_zenan)
print(cohen_kappa_score(list(le_sentiment.transform(sentiment_levels_bryan)), list(le_sentiment.transform(sentiment_levels_zenan))))

le_convincingness = LabelEncoder()
le_convincingness.fit(convincingness_levels_bryan + convincingness_levels_zenan)
print(cohen_kappa_score(list(le_convincingness.transform(convincingness_levels_bryan)), list(le_convincingness.transform(convincingness_levels_zenan))))




if __name__ == "__main__":
    with open("bryan_annotation_result.json") as f:
        bryan_results = json.load(f)
    print("Annotator Bryan:")
    single_behaviour(bryan_results)

    with open("zenan_annotation_result.json") as f:
        zenan_results = json.load(f)
    print("Annotator Zenan:")
    single_behaviour(zenan_results)
