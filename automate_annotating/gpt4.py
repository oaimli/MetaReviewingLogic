import openai
import jsonlines


openai.api_key = "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"

samples = []
with jsonlines.open("../../../HumanAnnotation/mrg_argument_tables/annotation/sampled_data.jsonl") as reader:
    for line in reader:
        samples.append(line)

prompt = open("prompt.txt").read()

for sample in samples:
    print(sample["paper_id"])
    print(sample.keys())
    source_documents = [sample["summary"]]
    for source_document in source_documents:
        try:
            output_dict = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt.replace("{{source_document}}", source_document)}
                ]
            )
            print(source_document)
            print(output_dict['choices'][0]['message']['content'])
        except:
            continue
    break



