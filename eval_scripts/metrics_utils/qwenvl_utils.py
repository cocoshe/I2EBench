import requests
# from retry import retry

def qwenvl_run(edited_path, question):

    print('in qwenvl_run')
    print('edited_path check:', edited_path)

    # QwenVL 本地 API 地址
    url = "http://localhost:3001/v1/chat/completions"

    # # 请求体格式，直接传本地图片路径
    # data = {
    #     "model": "Qwen/Qwen3-VL-8B-Instruct",
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": question},
    #                 {
    #                     "type": "image_url",
    #                     "image": edited_path,  # 本地文件路径
    #                 },
    #             ],
    #         }
    #     ],
    #     "max_tokens": 30,
    # }
    data = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question + '. Answer directly, no extra explanation'},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": edited_path
                        },
                    },
                ],
            }
        ],
        "max_tokens": 30,
    }
    response = requests.post(url, json=data).json()

    vlm_output = response["choices"][0]["message"]["content"]

    print(f'edited_path: {edited_path}')
    print('vlm_output:', vlm_output)
    return vlm_output

    # # 发送请求
    # try:
    #     response = requests.post(url, json=data)
    #     response.raise_for_status()
    # except Exception as e:
    #     print("Request error:", e)
    #     # return "NO"

    # # 打印原始返回（方便调试 QwenVL 的返回结构）
    # print("Raw response:", response.text)

    # # 解析 JSON
    # try:
    #     resp_json = response.json()
    #     # 如果返回结构和 GPT 接口一致
    #     if "choices" in resp_json and len(resp_json["choices"]) > 0:
    #         vlm_output = resp_json["choices"][0]["message"]["content"]
    #     else:
    #         # 如果格式不同，可以打印出来调试
    #         print("Unexpected response format:", resp_json)
    #         vlm_output = "NO"
    # except Exception as e:
    #     print("Parse response error:", e)
    #     # return "NO"
    # import pdb; pdb.set_trace()
    # print(f'edited_path: {edited_path}')
    # print('vlm_output:', vlm_output)

    # return vlm_output
