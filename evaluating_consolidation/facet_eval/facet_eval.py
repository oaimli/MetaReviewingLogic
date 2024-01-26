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