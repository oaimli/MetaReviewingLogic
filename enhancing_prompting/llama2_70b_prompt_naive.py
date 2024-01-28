import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


def predict_without_pipe(input_text, model_name="meta-llama/Llama-2-7b-chat-hf", max_predict_length=64,
                         min_predict_length=1, do_sample=True, top_p=0.95, num_beams=1, temperature=0.7):
    # TODO This code without pipeline for 70B cannot work, need to figure the problems out
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        padding_side="right",
        use_fast=True,
    )
    print("tokenizer bos", tokenizer.bos_token, tokenizer.bos_token_id)
    print("tokenizer eos", tokenizer.eos_token, tokenizer.eos_token_id)
    print("tokenizer pad", tokenizer.pad_token, tokenizer.pad_token_id)
    print("tokenizer unk", tokenizer.unk_token, tokenizer.unk_token_id)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.cleanup_cache_files()
    print(model.config)
    print(model.hf_device_map)

    # generate
    model.to("cuda")
    model.eval()
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
    print(predicted_summary)


def predict_with_pipe(input_text, model_name="meta-llama/Llama-2-70b-chat-hf", max_prediction_length=128, top_p=0.95):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    input_ids = tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        return_attention_mask=True
    ).input_ids
    max_input_length = len(input_ids[0])
    print(max_input_length)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        input_text,
        do_sample=True,
        top_p=top_p,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_input_length + max_prediction_length,
    )

    return sequences[0]['generated_text']


if __name__ == "__main__":
    input = "#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today? #Person2#: I found it would be a good idea to get a check-up. #Person1#: Yes, well, you haven't had one for 5 years. You should have one every year. #Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor? #Person1#: Well, the best way to avoid serious illnesses is to find out about them early. So try to come at least once a year for your own good. #Person2#: Ok. #Person1#: Let me see here. Your eyes and ears look fine. Take a deep breath, please. Do you smoke, Mr. Smith? #Person2#: Yes. #Person1#: Smoking is the leading cause of lung cancer and heart disease, you know. You really should quit. #Person2#: I've tried hundreds of times, but I just can't seem to kick the habit. #Person1#: Well, we have classes and some medications that might help. I'll give you more information before you leave. #Person2#: Ok, thanks doctor."
    target = "Mr. Smith's getting a check-up, and Doctor Hawkins advises him to have one every year. Hawkins'll give some information about their classes and medications to help Mr. Smith quit smoking."
    prompt_format = "<s>[INST] <<SYS>> Please write a summary for the conversation.<</SYS>> \nConversation:\n{input_text}\n [/INST] Summary:\n"
    input_text = prompt_format.format(input_text=input)
    # input = "What are the most famous universities in Australia?"
    # target = "UoM"
    # predict_without_pipe(input_text, model_name="meta-llama/Llama-2-70b-chat-hf")
    result = predict_with_pipe(input_text, model_name="meta-llama/Llama-2-70b-chat-hf", max_prediction_length=128)
    print(result)
