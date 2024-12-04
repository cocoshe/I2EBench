import json
from json import encoder
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

from metrics_utils.chatgpt_utils import chatgpt_run

PROJECT_ROOT = "/home/ma-user/work/mayiwei/yk/new_editbench/EditBench"
SRC_PATH = "EditData"
# DST_PATH = "EditResult"
# ORI_DST_PATH = "EditResult_ori"
EDIT_EVAL = 'EditEval_3nd'
EDIT_EVAL_ORI = 'EditEval_ori_3nd'

EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_3nd')
ORI_EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_ori_3nd')

if not os.path.exists(EVAL_PATH):
    os.mkdir(EVAL_PATH)
if not os.path.exists(ORI_EVAL_PATH):
    os.mkdir(ORI_EVAL_PATH)

HIGH_LEVEL_TASKS = [
    'Counting',                 # number
    'DirectionPerception',      # Y/N
    'ObjectRemoval',            # Y/N
    'Replacement',              # Y/N
    'BGReplacement',            # Y/N
    'ColorAlteration',          # A/B/C
    ####################################
    #################################### 'StyleAlteration',        # NOTE: calculate clip score, calculated in stage1, don't add here!
]

EDIT_MODELS = [
    'hive',
    'instructpix2pix',
    'magicbrush',
    'mgie',
    'instruct-diffusion',
    'any2pix',
    'iedit',
    'hqedit',
]

NUMBER_MODELS = [
    'Counting',                 # number
]

YES_NO_MODELS = [
    'DirectionPerception',      # Y/N
    'ObjectRemoval',            # Y/N
    'Replacement',              # Y/N
    'BGReplacement',            # Y/N
]

CHOICES_MODELS = [
    'ColorAlteration',          # A/B/C
]

for eval in [EDIT_EVAL_ORI, EDIT_EVAL]:
    for task_id, task in enumerate(HIGH_LEVEL_TASKS):
        task_path = os.path.join(PROJECT_ROOT, eval, task)
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        
        for model_id, model in enumerate(EDIT_MODELS):
            print('ChatGPT final judge on [{}] task({}/{}), with [{}] model({}/{}):'.format(task, task_id+1, len(HIGH_LEVEL_TASKS), model, model_id+1, len(EDIT_MODELS)))
            high_level_eval_out = {} 
            with open(os.path.join(task_path, model) + '.json', 'r') as f:
                MLLM_data = json.load(f)
            
            final_judge_out = {}
            for id, info in tqdm(MLLM_data.items()):
                final_judge_info = info
                final_judge, chatgpt_prompt = chatgpt_run(info['question'], info['evaluation']['VLM_judgement'], info['gt'])
                final_judge_info['final_judgement'] = final_judge

                print('-'*50)
                print('chatgpt_prompt: ', chatgpt_prompt)
                print('chatgpt final judgement: ', final_judge)
                print('-'*50)
                
                final_judge_out[id] = final_judge_info
            with open(os.path.join(task_path, model) + '.json', 'w') as f:
                print(f'save in {task_path} ')
                json.dump(final_judge_out, f, indent=4)
        
