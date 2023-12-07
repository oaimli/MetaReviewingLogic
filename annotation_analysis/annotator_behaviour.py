import json
import numpy as np
from rouge_score import rouge_scorer
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize


def annotated_documents(results):
    count_documents = 0
    count_judgements = 0
    for k, v in results.items():
        count_documents += len(v)
        for document in v:
            count_judgements += len(document["Annotated Judgements"])
    return count_documents / len(results.keys()), count_judgements / count_documents, count_judgements


def annotation_length(results):
    lens_content = []
    lens_sentiment = []
    for k, v in results.items():
        for document in v:
            # print(document["Document Title"])
            for judgement in document["Annotated Judgements"]:
                lens_content.append(len(judgement["Content Expression"].strip().split()))
                lens_sentiment.append(len(judgement["Sentiment Expression"].strip().split()))
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

    return pd.value_counts(criteria_facets, normalize=True), pd.value_counts(sentiment_expressers,
                                                                             normalize=True), pd.value_counts(
        sentiment_polarities, normalize=True), pd.value_counts(convincingnesses, normalize=True)


def single_behaviour(results):
    print("Average annotated documents per sample: ", annotated_documents(results)[0])
    print("All annotated judgements: ", annotated_documents(results)[2])
    print("Average annotated judgements per document: ", annotated_documents(results)[1])
    print("Average length of content expression: ", annotation_length(results)[0])
    print("Average length of sentiment expression: ", annotation_length(results)[1])
    print("Distribution of criteria facets: \n", distribution(results)[0])
    print("Distribution of sentiment expressers: \n", distribution(results)[1])
    print("Distribution of sentiment levels: \n", distribution(results)[2])
    print("Distribution of convincingnesses: \n", distribution(results)[3])


def character_level_agreement(annotation_data, results_1, results_2):
    content_1s = []
    content_2s = []
    sentiment_1s = []
    sentiment_2s = []
    all_1s = []
    all_2s = []

    for id, sample in annotation_data.items():
        print(id)
        result_1 = results_1[id]
        result_2 = results_2[id]

        meta_review = sample["meta_review"]
        meta_review_title = sample["meta_review_title"]

        meta_review_signal_content_1 = [0]*len(meta_review)
        meta_review_signal_sentiment_1 = [0] * len(meta_review)
        meta_review_signal_all_1 = [0] * len(meta_review)
        judgements_1 = []
        if result_1[0]["Document Title"] == meta_review_title:
            judgements_1 = result_1[0]["Annotated Judgements"]
        else:
            print("Error, meta_review_title")
        for judgement in judgements_1:
            content = judgement["Content Expression"]
            sentiment = judgement["Sentiment Expression"]
            start = 0
            while start >= 0:
                start = meta_review.find(content, start)
                if start != -1:
                    meta_review_signal_content_1[start: start + len(content)] = [1] * len(content)
                    meta_review_signal_all_1[start: start + len(content)] = [1] * len(content)
                    start += len(content)
            start = 0
            while start >= 0:
                start = meta_review.find(sentiment, start)
                if start != -1:
                    meta_review_signal_sentiment_1[start: start + len(sentiment)] = [1] * len(sentiment)
                    meta_review_signal_all_1[start: start + len(sentiment)] = [1] * len(sentiment)
                    start += len(sentiment)
        content_1s.extend(meta_review_signal_content_1)
        sentiment_1s.extend(meta_review_signal_sentiment_1)
        all_1s.extend(meta_review_signal_all_1)

        meta_review_signal_content_2 = [0] * len(meta_review)
        meta_review_signal_sentiment_2 = [0] * len(meta_review)
        meta_review_signal_all_2 = [0] * len(meta_review)
        judgements_2 = []
        if result_2[0]["Document Title"] == meta_review_title:
            judgements_2 = result_2[0]["Annotated Judgements"]
        else:
            print("Error, meta_review_title")
        for judgement in judgements_2:
            content = judgement["Content Expression"]
            sentiment = judgement["Sentiment Expression"]
            start = 0
            while start >= 0:
                start = meta_review.find(content, start)
                if start != -1:
                    meta_review_signal_content_2[start: start + len(content)] = [1] * len(content)
                    meta_review_signal_all_2[start: start + len(content)] = [1] * len(content)
                    start += len(content)
            start = 0
            while start >= 0:
                start = meta_review.find(sentiment, start)
                if start != -1:
                    meta_review_signal_sentiment_2[start: start + len(sentiment)] = [1] * len(sentiment)
                    meta_review_signal_all_2[start: start + len(sentiment)] = [1] * len(sentiment)
                    start += len(sentiment)
        content_2s.extend(meta_review_signal_content_2)
        sentiment_2s.extend(meta_review_signal_sentiment_2)
        all_2s.extend(meta_review_signal_all_2)

        for review in sample["reviews"]:
            title = review["title"]
            original_document = review["comment"]

            judgements_1_tmp = []
            for document in result_1:
                if title == document["Document Title"]:
                    judgements_1_tmp = document["Annotated Judgements"]
                    break
            if len(judgements_1_tmp) == 0:
                print("No judgements", title)
            judgements_2_tmp = []
            for document in result_2:
                if title == document["Document Title"]:
                    judgements_2_tmp = document["Annotated Judgements"]
                    break
            if len(judgements_2_tmp) == 0:
                print("No judgements", title)

            signal_content_1 = [0] * len(original_document)
            signal_sentiment_1 = [0] * len(original_document)
            signal_all_1 = [0] * len(original_document)
            for judgement in judgements_1_tmp:
                content = judgement["Content Expression"]
                sentiment = judgement["Sentiment Expression"]
                start = 0
                while start >= 0:
                    start = original_document.find(content, start)
                    if start != -1:
                        signal_content_1[start: start + len(content)] = [1] * len(content)
                        signal_all_1[start: start + len(content)] = [1] * len(content)
                        start += len(content)
                start = 0
                while start >= 0:
                    start = original_document.find(sentiment, start)
                    if start != -1:
                        signal_sentiment_1[start: start + len(sentiment)] = [1] * len(sentiment)
                        signal_all_1[start: start + len(sentiment)] = [1] * len(sentiment)
                        start += len(sentiment)
            content_1s.extend(signal_content_1)
            sentiment_1s.extend(signal_sentiment_1)
            all_1s.extend(signal_all_1)

            signal_content_2 = [0] * len(original_document)
            signal_sentiment_2 = [0] * len(original_document)
            signal_all_2 = [0] * len(original_document)
            for judgement in judgements_2_tmp:
                content = judgement["Content Expression"]
                sentiment = judgement["Sentiment Expression"]
                start = 0
                while start >= 0:
                    start = original_document.find(content, start)
                    if start != -1:
                        signal_content_2[start: start + len(content)] = [1] * len(content)
                        signal_all_2[start: start + len(content)] = [1] * len(content)
                        start += len(content)
                start = 0
                while start >= 0:
                    start = original_document.find(sentiment, start)
                    if start != -1:
                        signal_sentiment_2[start: start + len(sentiment)] = [1] * len(sentiment)
                        signal_all_2[start: start + len(sentiment)] = [1] * len(sentiment)
                        start += len(sentiment)
            content_2s.extend(signal_content_2)
            sentiment_2s.extend(signal_sentiment_2)
            all_2s.extend(signal_all_2)

    print("#### Hightlight correlation, content, character level")
    a = np.array(content_1s)
    b = np.array(content_2s)
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))

    print("#### Hightlight correlation, sentiment, character level")
    a = np.array(sentiment_1s)
    b = np.array(sentiment_2s)
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))

    print("#### Hightlight correlation, content+sentiment, character level")
    a = np.array(all_1s)
    b = np.array(all_2s)
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))


def annotator_agreement(results_1, results_2):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    with open("../annotation_data/annotation_data_small.json") as f:
        annotation_data_tmp = json.load(f)
    annotation_data = {}
    for key in annotation_data_tmp.keys():
        if key in results_1.keys() and key in results_2.keys():
            annotation_data[key] = annotation_data_tmp[key]
    assert len(annotation_data.keys()) == len(results_1.keys()) == len(results_2.keys())

    character_level_agreement(annotation_data, results_1, results_2)

    # Split documents into sentences, and anchor judgements to corresponding sentences
    for id in annotation_data.keys():
        original_data = annotation_data[id]
        meta_review = original_data["meta_review"]
        reviews = original_data["reviews"]
        reviews_new = []
        for review in reviews:
            review["sentences"] = sent_tokenize(review["comment"])
            reviews_new.append(review)
        annotation_data[id]["reviews"] = reviews_new
        annotation_data[id]["meta_review_sentences"] = sent_tokenize(meta_review)

        result_documents_1 = results_1[id]
        result_documents_1_new = []
        for i, result_document_1 in enumerate(result_documents_1):
            title = result_document_1["Document Title"]
            if i == 0 and title == original_data["meta_review_title"]:
                original_sentences = original_data["meta_review_sentences"]
            else:
                for review in annotation_data[id]["reviews"]:
                    if review["title"] == title:
                        original_sentences = review["sentences"]
                        break
            judgements = result_document_1["Annotated Judgements"]
            # print(judgements)
            # print(original_sentences)

            sentences = []
            for judgement in judgements:
                tmp = judgement["Content Expression"] + " " + judgement["Sentiment Expression"]
                rouges = []
                for sentence in original_sentences:
                    scores = scorer.score(tmp, sentence)
                    rouges.append(scores["rouge2"].fmeasure + scores["rouge1"].fmeasure + scores["rougeLsum"].fmeasure)
                sentences.append(rouges.index(max(rouges)))

            # offset = len(sentences) - len(set(sentences))
            # if offset > 0:
            #     print(sentences)
            result_document_1["Sentences"] = sentences
            result_documents_1_new.append(result_document_1)
        results_1[id] = result_documents_1_new

        result_documents_2 = results_2[id]
        result_documents_2_new = []
        for i, result_document_2 in enumerate(result_documents_2):
            title = result_document_2["Document Title"]
            if i == 0 and title == original_data["meta_review_title"]:
                original_sentences = original_data["meta_review_sentences"]
            else:
                for review in annotation_data[id]["reviews"]:
                    if review["title"] == title:
                        original_sentences = review["sentences"]
                        break
            judgements = result_document_2["Annotated Judgements"]

            sentences = []
            for judgement in judgements:
                tmp = judgement["Content Expression"] + " " + judgement["Sentiment Expression"]
                rouges = []
                for sentence in original_sentences:
                    scores = scorer.score(tmp, sentence)
                    rouges.append(scores["rouge2"].fmeasure + scores["rouge1"].fmeasure + scores["rougeLsum"].fmeasure)
                sentences.append(rouges.index(max(rouges)))

            result_document_2["Sentences"] = sentences
            result_documents_2_new.append(result_document_2)
        results_2[id] = result_documents_2_new

    # Get shared judgements, based on annotation results
    judgements_bryan_share = []
    judgements_zenan_share = []
    for id in results_1.keys():
        documents_1 = results_1[id]
        documents_2 = results_2[id]
        for document_1 in documents_1:
            title_1 = document_1["Document Title"]
            judgements_1 = document_1["Annotated Judgements"]
            sentences_1 = document_1["Sentences"]
            judgements_2 = []
            sentences_2 = []
            for document_2 in documents_2:
                title_2 = document_2["Document Title"]
                if title_1 == title_2:
                    judgements_2 = document_2["Annotated Judgements"]
                    sentences_2 = document_2["Sentences"]
                    break
            if len(judgements_2) > 0:
                all_indexes = set(sentences_1).intersection(set(sentences_2))
                for index in all_indexes:
                    if sentences_1.count(index) == 1 and sentences_2.count(index) == 1:
                        for judgement_1, sentence_1 in zip(judgements_1, sentences_1):
                            if sentence_1 == index:
                                judgements_bryan_share.append(judgement_1)
                                break
                        for judgement_2, sentence_2 in zip(judgements_2, sentences_2):
                            if sentence_2 == index:
                                judgements_zenan_share.append(judgement_2)
                                break
                    if sentences_1.count(index) > 1 or sentences_2.count(index) > 1:
                        judgements_1_tmp = []
                        for judgement_1, sentence_1 in zip(judgements_1, sentences_1):
                            if sentence_1 == index:
                                judgements_1_tmp.append(judgement_1)
                        judgements_2_tmp = []
                        for judgement_2, sentence_2 in zip(judgements_2, sentences_2):
                            if sentence_2 == index:
                                judgements_2_tmp.append(judgement_2)

                        for judgement_1 in judgements_1_tmp:
                            tmp_1 = judgement_1["Content Expression"] + " " + judgement_1["Sentiment Expression"]
                            for judgement_2 in judgements_2_tmp:
                                tmp_2 = judgement_2["Content Expression"] + " " + judgement_2["Sentiment Expression"]
                                scores = scorer.score(tmp_1, tmp_2)
                                s = scores["rouge2"].fmeasure + scores["rouge1"].fmeasure + scores[
                                    "rougeLsum"].fmeasure
                                if s > 1.6:
                                    judgements_bryan_share.append(judgement_1)
                                    judgements_zenan_share.append(judgement_2)
    print("All shared judgements: ", len(judgements_bryan_share), len(judgements_zenan_share))

    # Correlation on highlight, sentence level, based on original annotation data
    print("###### Correlation on judgement, sentence level")
    actions_all_bryan = []
    actions_all_zenan = []
    for id, sample in annotation_data.items():
        result_1 = results_1[id]
        result_2 = results_2[id]

        meta_review_sentences = sample["meta_review_sentences"]
        for sentence_id, sentence in enumerate(meta_review_sentences):
            result_1_meta_review_judgements = result_1[0]["Sentences"]
            result_2_meta_review_judgements = result_2[0]["Sentences"]
            if sentence_id in result_1_meta_review_judgements:
                actions_all_bryan.append(1)
            else:
                actions_all_bryan.append(0)
            if sentence_id in result_2_meta_review_judgements:
                actions_all_zenan.append(1)
            else:
                actions_all_zenan.append(0)

        for review in sample["reviews"]:
            title = review["title"]
            annotation_sentences = review["sentences"]

            judgement_sentences_1 = []
            for document in result_1:
                if document["Document Title"] == title:
                    judgement_sentences_1 = document["Sentences"]
                    break
            for sentence_id, sentence in enumerate(annotation_sentences):
                if sentence_id in judgement_sentences_1:
                    actions_all_bryan.append(1)
                else:
                    actions_all_bryan.append(0)

            judgement_sentences_2 = []
            for document in result_2:
                if document["Document Title"] == title:
                    judgement_sentences_2 = document["Sentences"]
                    break
            for sentence_id, sentence in enumerate(annotation_sentences):
                if sentence_id in judgement_sentences_2:
                    actions_all_zenan.append(1)
                else:
                    actions_all_zenan.append(0)
    a = np.array(actions_all_bryan)
    b = np.array(actions_all_zenan)
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))

    # Correlation on different aspects
    criteria_facet_same = []
    sentiment_level_same = []
    sentiment_expresser_same = []
    convincingness_same = []
    sentiment_polarity_same = []

    criteria_facets_bryan = []
    criteria_facets_zenan = []
    sentiment_expressers_bryan = []
    sentiment_expressers_zenan = []
    sentiment_levels_bryan = []
    sentiment_levels_zenan = []
    convincingness_levels_bryan = []
    convincingness_levels_zenan = []
    sentiment_polarities_bryan = []
    sentiment_polarities_zenan = []
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

        criteria_facets_bryan.append(criteria_facet_bryan)
        criteria_facets_zenan.append(criteria_facet_zenan)

        sentiment_expressers_bryan.append(sentiment_expresser_bryan)
        sentiment_expressers_zenan.append(sentiment_expresser_zenan)

        sentiment_levels_bryan.append(sentiment_polarity_bryan)
        sentiment_levels_zenan.append(sentiment_polarity_zenan)

        convincingness_levels_bryan.append(convincingness_bryan)
        convincingness_levels_zenan.append(convincingness_zenan)

        if "ositive" in sentiment_polarity_bryan:
            sentiment_polarities_bryan.append("Positive")
        if "egative" in sentiment_polarity_bryan:
            sentiment_polarities_bryan.append("Negative")
        if "ositive" in sentiment_polarity_zenan:
            sentiment_polarities_zenan.append("Positive")
        if "egative" in sentiment_polarity_zenan:
            sentiment_polarities_zenan.append("Negative")

    print("Shared criteria_facet in judgements: ", len(criteria_facet_same))
    print(pd.value_counts(criteria_facet_same, normalize=True))
    print("Shared sentiment_level in judgements: ", len(sentiment_level_same))
    print(pd.value_counts(sentiment_level_same, normalize=True))
    print("Shared sentiment_expresser in judgements: ", len(sentiment_expresser_same))
    print(pd.value_counts(sentiment_expresser_same, normalize=True))
    print("Shared convincingness in judgements: ", len(convincingness_same))
    print(pd.value_counts(convincingness_same, normalize=True))
    print("Shared sentiment_polarity in judgements: ", len(sentiment_polarity_same))
    print(pd.value_counts(sentiment_polarity_same, normalize=True))

    print("###### Correlation on criteria facets:")
    le_criteria_facets = LabelEncoder()
    le_criteria_facets.fit(criteria_facets_bryan + criteria_facets_zenan)
    a = np.array(list(le_criteria_facets.transform(criteria_facets_bryan)))
    b = np.array(list(le_criteria_facets.transform(criteria_facets_zenan)))
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))

    print("###### Correlation on sentiment expressers:")
    le_sentiment_expressers = LabelEncoder()
    le_sentiment_expressers.fit(sentiment_expressers_bryan + sentiment_expressers_zenan)
    a = np.array(list(le_sentiment_expressers.transform(sentiment_expressers_bryan)))
    b = np.array(list(le_sentiment_expressers.transform(sentiment_expressers_zenan)))
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))

    print("###### Correlation on sentiment levels:")
    le_sentiment_levels = LabelEncoder()
    le_sentiment_levels.fit(sentiment_levels_bryan + sentiment_levels_zenan)
    a = np.array(list(le_sentiment_levels.transform(sentiment_levels_bryan)))
    b = np.array(list(le_sentiment_levels.transform(sentiment_levels_zenan)))
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))

    print("###### Correlation on convincingness levels:")
    le_convincingness = LabelEncoder()
    le_convincingness.fit(convincingness_levels_bryan + convincingness_levels_zenan)
    a = np.array(list(le_convincingness.transform(convincingness_levels_bryan)))
    b = np.array(list(le_convincingness.transform(convincingness_levels_zenan)))
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))

    print("###### Correlation on sentiment polarities:")
    le_sentiment_polarities = LabelEncoder()
    le_sentiment_polarities.fit(sentiment_polarities_bryan + sentiment_polarities_zenan)
    a = np.array(list(le_sentiment_polarities.transform(sentiment_polarities_bryan)))
    b = np.array(list(le_sentiment_polarities.transform(sentiment_polarities_zenan)))
    print("Cohen Kappa: ", cohen_kappa_score(a, b))
    print("Kendall Tau: ", stats.kendalltau(a, b))
    print("Spearman: ", stats.spearmanr(a, b))
    print("Pearson: ", stats.pearsonr(a, b))


if __name__ == "__main__":
    with open("bryan_annotation_result.json") as f:
        bryan_results = json.load(f)
    print("################ Annotator Bryan: ################")
    single_behaviour(bryan_results)

    with open("zenan_annotation_result.json") as f:
        zenan_results = json.load(f)
    print("################ Annotator Zenan: ################")
    single_behaviour(zenan_results)

    print("################ Annotator Agreement: ################")
    annotator_agreement(bryan_results, zenan_results)
