import os
import pandas
import json

annotation_folder = "../../HumanAnnotation/mrg_judgement"
bryan_result_folder = "annotation/done_1108/meta-review-annotation-bryan"
zenan_result_folder = "annotation/done_1108/meta-review-annotation-zenan"

bryan_files = os.listdir(os.path.join(annotation_folder, bryan_result_folder))
zenan_files = os.listdir(os.path.join(annotation_folder, zenan_result_folder))

print(len(bryan_files), len(zenan_files))
assert len(set(bryan_files).difference(set(zenan_files))) == 0


# load results from Excel
def load_results(annotation_folder, result_folder):
    results = {}
    for file in bryan_files:
        documents = []
        sheet = pandas.read_excel(io=os.path.join(annotation_folder, result_folder, file), sheet_name=0, header=0)
        sheet_dict = sheet.to_dict('index')
        items = []
        document_title = ""
        for k in sheet_dict.keys():
            row = sheet_dict[k]
            if pandas.notna(row["Content Expression"]) and pandas.notna(row["Sentiment Expression"]):
                if pandas.notna(row["Document Title"]):
                    document_title = row["Document Title"].strip()
                del row["Document Title"]
                items.append(row)

            if k + 1 < len(sheet_dict):
                row_next = sheet_dict[k + 1]
                if pandas.notna(row_next["Document Title"]) and len(items) > 0:
                    documents.append({"Document Title": document_title, "Annotated Judgements": items})
                    items = []
                if not pandas.notna(row["Content Expression"]) and not pandas.notna(row["Sentiment Expression"]) and len(items) > 0:
                    documents.append({"Document Title": document_title, "Annotated Judgements": items})
                    items = []
        # print(len(result))
        results[file[:-5]] = documents
    return results


bryan_results = load_results(annotation_folder, bryan_result_folder)
with open("bryan_annotation_result.json", "w") as f:
    json.dump(bryan_results, f, indent=4)
zenan_results = load_results(annotation_folder, zenan_result_folder)
with open("zenan_annotation_result.json", "w") as f:
    json.dump(zenan_results, f, indent=4)
