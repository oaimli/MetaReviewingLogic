from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import json
from tqdm import tqdm
import os
import torch


def predicting(model, tokenizer, input_text, max_predict_length=512, min_predict_length=1, do_sample=True, top_p=0.95, num_beams=1, temperature=0.7):
    print(input_text)
    input_dict = tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=True
    )
    input_ids = input_dict.input_ids
    attention_mask = input_dict.attention_mask
    output_ids = model.generate(
        input_ids=input_ids.to("cuda"),
        attention_mask=attention_mask.to("cuda"),
        max_length=len(input_ids[0]) + max_predict_length,
        min_length=len(input_ids[0]) + min_predict_length,
        do_sample=do_sample,
        top_p=top_p,
        num_beams=num_beams,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    predicted_result = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    print(predicted_result)
    return predicted_result


if __name__ == "__main__":
    # load model and tokenizer
    # load model and tokenizer
    model_name = "meta-llama/Llama-2-70b-chat-hf"

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=4096,
        padding_side="right",
        use_fast=True,
    )
    print("tokenizer bos", tokenizer.bos_token, tokenizer.bos_token_id)
    print("tokenizer eos", tokenizer.eos_token, tokenizer.eos_token_id)
    print("tokenizer pad", tokenizer.pad_token, tokenizer.pad_token_id)
    print("tokenizer unk", tokenizer.unk_token, tokenizer.unk_token_id)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.cleanup_cache_files()
    print(model.config)
    print(model.hf_device_map)
    model.eval()

    # load the prompt
    prompt_format_summarizing_each_facet = open("prompts/prompt_summarizing_each_facet.txt").read()
    prompt_format_aggregating_sub_summaries = open("prompts/prompt_aggregating_sub_summaries.txt").read()

    with open("test_data.json") as f:
        test_samples = json.load(f)

    source_judgements = {}
    source_judgements_folder = "../evaluating_consolidation/fusion_eval_judgements_tmp/test_data"
    file_names = os.listdir(source_judgements_folder)
    for file_name in file_names:
        with open(os.path.join(source_judgements_folder, file_name)) as f:
            source_judgements.update(json.load(f))
    print(len(source_judgements))

    results = {}
    for key, sample in tqdm(test_samples.items()):
        judgements = source_judgements[key]
        organized = {}
        for judgement in judgements:
            criteria_facet = judgement["Criteria Facet"]
            tmp = organized.get(criteria_facet, [])
            tmp.append(judgement)
            organized[criteria_facet] = tmp

        sub_summaries = []
        for k, v in organized.items():
            source_judgements_text = []
            for source_judgement in v:
                source_judgements_text.append(str(source_judgement))
            tmp = prompt_format_summarizing_each_facet.replace("{{input_judgements}}", "\n".join(source_judgements_text)).replace(
                "{{criteria_facet}}", k)
            # print(prompt_format)
            sub_summary = predicting(model, tokenizer, tmp)
            sub_summaries.append(k + "\n" + sub_summary)

        result = predicting(model, tokenizer, prompt_format_aggregating_sub_summaries.replace("{{input_sub_summaries}}", "\n".join(sub_summaries)))
        results[key] = {"generation": result}

    print(len(results))
    with open("results/generation_llama2_70b_pipeline_ours.json", "w") as f:
        json.dump(results, f, indent=4)
