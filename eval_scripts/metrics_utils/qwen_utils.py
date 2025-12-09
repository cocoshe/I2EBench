# import openai
# import base64
# from retry import retry
import requests



def qwen_run(question, vlm_judgement, gt_answer):
    url = "http://localhost:3001/v1/chat/completions"
    chatgpt_prompt = f'''
    Question: {question}
    Correct Answer: {gt_answer}
    Machine Response: {vlm_judgement}
    Is the machine's answer correct? Answer yes or no. No extra explanation
    '''
    data = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": chatgpt_prompt},
                    
                ],
            }
        ],
        "max_tokens": 30,
    }
    response = requests.post(url, json=data).json()

    vlm_output = response["choices"][0]["message"]["content"]


    return vlm_output, chatgpt_prompt




