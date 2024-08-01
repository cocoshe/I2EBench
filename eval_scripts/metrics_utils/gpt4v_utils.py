import openai
import base64
from retry import retry

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@retry(delay=1, backoff=2, max_delay=4)
def gpt4_run(edited_path, question):
    print('in gpt4_run')

    base64_image = encode_image(edited_path)

    msg=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": question,
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
            ],
        }
    ]

    model = "gpt-4-vision-preview"
    # Use your own GPT4 API configuration
    openai.base_url = 'http://openai.infly.tech/v1/'
    openai.api_key = 'no-modify'
    extra = {}

    stream = False
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=msg,
            extra_body=extra,
            extra_headers={'apikey':'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'},
            stream=stream,
            max_tokens=30,
        )
    except openai.APIStatusError as e:
        print('edited_path: ', edited_path)
        print(f"headers:{e.response.headers}, resp:{e.response.content}")


    vlm_output = response.choices[0].message.content

    return vlm_output



