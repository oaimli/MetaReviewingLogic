import openai
import time


openai.api_key = "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"

prompt_facet = open("prompts/prompt_facet_old.txt").read()
prompt_expresser = open("prompts/prompt_expresser_old.txt").read()
prompt_convincingness = open("prompts/prompt_convincingness_old.txt").read()
prompt_polarity = open("prompts/prompt_polarity_old.txt").read()

try:
    output = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt_polarity},
            {"role": "system", "content": "Please help to improve the above prompt to be better and fullfill the underlying steps to do this annotation task in the origianl prompt."}
        ]
    )
    print(output['choices'][0]['message']['content'])
except Exception as e:
    print(e)
    if ("limit" in str(e)):
        time.sleep(2)