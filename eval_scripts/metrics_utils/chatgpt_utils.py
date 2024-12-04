import openai
import base64
from retry import retry


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@retry(delay=1, backoff=2, max_delay=4)
def chatgpt_run(question, vlm_judgement, gt_answer):
    chatgpt_prompt = f'''
Question: {question}
Correct Answer: {gt_answer}
Machine Response: {vlm_judgement}
Is the machine's answer correct? Answer yes or no.
'''

    msg=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": chatgpt_prompt,
            },
            ],
        }
    ]

    model = "gpt-4-0125-preview"

    openai.base_url = 'input your base url'
    openai.api_key = 'no-modify'
    extra = {}

    stream = False
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=msg,
            extra_body=extra,
            extra_headers={'apikey':'xxxxxxxxxxxxxxxxxxxxxx'},
            stream=stream,
            max_tokens=30,
        )
    except openai.APIStatusError as e:
        print(f"headers:{e.response.headers}, resp:{e.response.content}")


    vlm_output = response.choices[0].message.content

    return vlm_output, chatgpt_prompt



