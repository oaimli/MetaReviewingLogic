import json

with open("bryan_annotation_result.json") as f:
    bryan_results = json.load(f)

with open("zenan_annotation_result.json") as f:
    zenan_results = json.load(f)


sample_indexes = bryan_results.keys()
for sample_index in sample_indexes:
    sample_bryan = bryan_results[bryan_results]
    sample_zenan = zenan_results[zenan_results]


