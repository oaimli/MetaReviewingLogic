from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import json


def predict(model, tokenizer, input_text, max_predict_length=512, min_predict_length=1, do_sample=True, top_p=0.95, num_beams=1, temperature=0.7):
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
    predicted_summary = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=False)
    return predicted_summary


if __name__ == "__main__":
    # load model and tokenizer
    model_in_use = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(
        model_in_use,
        model_max_length=4096,
        padding_side="right",
        use_fast=True,
    )
    print("tokenizer bos", tokenizer.bos_token, tokenizer.bos_token_id)
    print("tokenizer eos", tokenizer.eos_token, tokenizer.eos_token_id)
    print("tokenizer pad", tokenizer.pad_token, tokenizer.pad_token_id)
    print("tokenizer unk", tokenizer.unk_token, tokenizer.unk_token_id)

    model = AutoModelForCausalLM.from_pretrained(model_in_use, trust_remote_code=True)
    print(model.config)
    model.to("cuda")
    model.eval()

    # load the prompt
    prompt_format = open("prompts/prompt_naive.txt").read()

    with open("test_data.json") as f:
        test_samples = json.load(f)

    results = {}
    for key, sample in test_samples.items():
        input_texts = []
        for review in sample["reviews"]:
            if review["writer"] == "official_reviewer" and review["reply_to"] == sample["paper_id"]:
                input_texts.append(review["comment"])
        result = predict(model, tokenizer, prompt_format.replace("{{input_documents}}", "\n".join(input_texts)))
        results[key] = {"generation": result}
        break

    print(len(results))
    with open("results/generation_llama2_7b_prompt_naive.json", "w") as f:
        json.dump(results, f, indent=4)
