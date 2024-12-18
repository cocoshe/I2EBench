import openai
import base64
from retry import retry

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# @retry(delay=1, backoff=2, max_delay=4)
def gpt4_run(edited_path, question):
    print('in gpt4_run')
    print('edited_path check: ', edited_path)
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

    # model = "gpt-4-vision-preview"
    model = "gpt-4o"

    openai.base_url = 'input your base url'

    openai.api_key = 'no-modify'
    extra = {}



    stream = False


    # try:
    #     response = openai.chat.completions.create(
    #         model=model,
    #         messages=msg,
    #         extra_body=extra,
    #         extra_headers={'apikey':'xxxxxxxxxxxxx'},
    #         stream=stream,
    #         max_tokens=30,
    #     )
    # except Exception as e:
    #     print('edited_path: ', edited_path)
    #     # print(f"headers:{e.response.headers}, resp:{e.response.content}")
    #     print(e)
    #     print("ERROR!")
    #     resp = "NO"
    #     return resp


    response = openai.chat.completions.create(
        model=model,
        messages=msg,
        extra_body=extra,
        extra_headers={'apikey':'xxxxxxxxxxxxx'},
        stream=stream,
        max_tokens=30,
    )
    print('edited_path: ', edited_path)
    # print(f"headers:{e.response.headers}, resp:{e.response.content}")
    # print(e)




    vlm_output = response.choices[0].message.content
    print(f'edited_path: {edited_path}')
    print('vlm_output: ', vlm_output)

    return vlm_output



