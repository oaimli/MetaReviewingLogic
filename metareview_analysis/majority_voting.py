# Validate meta-review is not always following majority voting
import jsonlines
import numpy as np

all_samples = []
with jsonlines.open("../annotation_data/peersum_all.json") as reader:
    for line in reader:
        all_samples.append(line)

results = []
acceptances = []
for sample in all_samples:
    reviews = sample["reviews"]
    ratings = []
    for review in reviews:
        rating = review["rating"]
        if rating > 0:
            ratings.append(rating)

    count_positive = 0
    count_negative = 0
    for rating in ratings:
        if rating >= 5:
            count_positive += 1
        else:
            count_negative += 1
    if count_positive > count_negative:
        results.append(1)
    else:
        results.append(0)

    tmp = sample["paper_acceptance"].lower()
    if "accep" in tmp or "poster" in tmp or "workshop" in tmp or "oral" in tmp or "spotlight" in tmp:
        # if max(ratings) < 5:
        #     print(tmp)
        #     print(ratings, sample["paper_id"])
        acceptances.append(1)
    elif "rejec" in tmp:
        acceptances.append(0)
    else:
        print(tmp)

consistent = 0
for result, acceptance, sample in zip(results, acceptances, all_samples):
    if result == acceptance:
        consistent += 1
    else:
        print(sample["paper_id"])
print(consistent/len(results))