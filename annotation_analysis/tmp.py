# In every sample, the title of each document is unique
import json

with open("bryan_annotation_result.json") as f:
    bryan_results = json.load(f)

with open("zenan_annotation_result.json") as f:
    zenan_results = json.load(f)

for id in bryan_results.keys():
    # print(id)
    bryan_documents = bryan_results[id]

    titles = []
    for document in bryan_documents:
        title = document["Document Title"]
        if title in titles:
            print("bryan", title)
        else:
            titles.append(document["Document Title"])

    zenan_documents = zenan_results[id]
    titles = []
    for document in zenan_documents:
        title = document["Document Title"]
        if title in titles:
            print("zenan", title)
        else:
            titles.append(document["Document Title"])

