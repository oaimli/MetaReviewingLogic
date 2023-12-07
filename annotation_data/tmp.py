# import jsonlines
# import json
#
# with open("../annotation_analysis/bryan_annotation_result.json") as f:
#     bryan_results = json.load(f)
# with open("../annotation_analysis/zenan_annotation_result.json") as f:
#     zenan_results = json.load(f)
#
# with open("../annotation_data/annotation_data_small.json") as f:
#     annotation_data = json.load(f)
#
# for id, documents in bryan_results.items():
#     print(id)
#     sample = annotation_data[id]
#
#     document_titles = []
#     document_titles.append(sample["meta_review_title"])
#     for review in sample["reviews"]:
#         document_titles.append(review["title"])
#
#     for document in documents:
#         if document["Document Title"] not in document_titles:
#             print(document)
#
# for id, documents in zenan_results.items():
#     print(id)
#     sample = annotation_data[id]
#
#     document_titles = []
#     document_titles.append(sample["meta_review_title"])
#     for review in sample["reviews"]:
#         document_titles.append(review["title"])
#
#     for document in documents:
#         if document["Document Title"] not in document_titles:
#             print(document)


import re

line="this hdr-biz model args= server server"
content = [0]*len(line)
signal_all = [0]*len(line)
print(line.find("server1", 0))

start = 0
while start >= 0:
    print(start)
    start = line.find("server", start)
    content[start: start + len("server")] = [1] * len("server")
    signal_all[start: start + len("server")] = [1] * len("server")
    if start != -1:
        start += len("server")
print(content)
print(signal_all)


