import json
from json import encoder
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

from metrics_utils.clip_utils import run_clip
from metrics_utils.gpt4v_utils import gpt4_run


PROJECT_ROOT = "/home/ma-user/work/mayiwei/yk/new_editbench/EditBench"
# SRC_PATH = "EditData"
# DST_PATH = "EditResult"
# ORI_DST_PATH = "EditResult_ori"

SRC_JSON_PATH = "EditData"
# SRC_PATH = "EditResult"
# DST_PATH = "EditResult_3nd"
SRC_PATH = "EditResult_ori"
DST_PATH = "EditResult_ori_3nd"

# EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval')
# ORI_EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_ori')

# EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_3nd')
EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_ori_3nd')


if not os.path.exists(EVAL_PATH):
    os.mkdir(EVAL_PATH)
# if not os.path.exists(ORI_EVAL_PATH):
#     os.mkdir(ORI_EVAL_PATH)

HIGH_LEVEL_TASKS = [
    'Counting',                 # number
    'DirectionPerception',      # Y/N
    'ObjectRemoval',            # Y/N
    'Replacement',              # Y/N
    'BGReplacement',            # Y/N
    'ColorAlteration',          # A/B/C
    ###################################
    'StyleAlteration',          # clip score
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

def calc_metrics(task, edit_model, image_name, dst, input_instruction, gt_answer, edit_prompt):
    # gt_path = os.path.join(PROJECT_ROOT, SRC_PATH, task, 'input', image_name)
    gt_path = os.path.join(PROJECT_ROOT, SRC_PATH, task, edit_model, image_name)
    edited_path = os.path.join(PROJECT_ROOT, dst, task, edit_model, image_name)

    print('before gpt4')
    vlm_output = gpt4_run(edited_path, input_instruction)
    print('-'*100)
    print('input_path: ', gt_path)
    print('edited_path: ', edited_path)
    print('edit_prompt: ', edit_prompt)
    print('question: ', input_instruction)
    print('gt_answer: ', gt_answer)
    print('vlm_output: ', vlm_output)
    print('-'*100)
    print('after gpt4')


    return vlm_output

def editedImg_caption_alignment(edit_model, image_name, dst, edited_caption):
    edited_path = os.path.join(PROJECT_ROOT, dst, 'StyleAlteration', edit_model, image_name)
    return run_clip(edited_caption, edited_path)



# for dst, val in zip([ORI_DST_PATH,DST_PATH], [ORI_EVAL_PATH,EVAL_PATH]):
for dst, val in zip([DST_PATH], [EVAL_PATH]):
    for task_id, task in enumerate(HIGH_LEVEL_TASKS):
        task_path = os.path.join(PROJECT_ROOT, SRC_PATH, task)
        if not os.path.exists(task_path):
            os.mkdir(task_path)

        # path = os.path.join(PROJECT_ROOT, SRC_JSON_PATH, task, task + '_3nd.json')
        path = os.path.join(PROJECT_ROOT, SRC_JSON_PATH, task, task + '_3nd.json')
        with open(path, 'r') as f:
            data = json.load(f)
        for model_id, edit_model in enumerate(EDIT_MODELS):
            print('Evaluating on [{}] task({}/{}), with [{}] model({}/{}):'.format(task, task_id+1, len(HIGH_LEVEL_TASKS), edit_model, model_id+1, len(EDIT_MODELS)))
            high_level_eval_out = {}
            for id, info in data.items():
                print(f'id: {int(id)+1}/{len(data)}, task: {task}, model: {edit_model}')
                print(id, info)
                image_name = info['image']
                dataset = task
                # if dst == ORI_DST_PATH:
                #     prompt = info['ori_exp']
                # elif dst == DST_PATH:
                #     prompt = info['div_exp']
                # else:
                #     raise ValueError('Something wrong here!')
                # prompt = info['div_exp']
                prompt = info['ori_exp']
                evaluation = {}

                if task == 'StyleAlteration':
                    evaluation['clip_score'] = editedImg_caption_alignment(edit_model, image_name, dst, info['style'])
                else:  # high level vlm
                    question = info['Evaluation']
                    prepared_prompt = question
                    evaluation['VLM_judgement'] = calc_metrics(task, edit_model, image_name, dst, prepared_prompt, info['Answer'], edit_prompt=prompt)

                sample = {}
                sample['image'] = image_name
                sample['dataset'] = dataset
                sample['prompt'] = prompt
                sample['evaluation'] = evaluation
                if task != 'StyleAlteration':
                    sample['question'] = question
                    sample['gt'] = info['Answer']
                    sample['type'] = info['type']

                high_level_eval_out[id] = sample

            
            task_path = os.path.join(val, task)
            if not os.path.exists(task_path):
                os.mkdir(task_path)

            save_path = os.path.join(task_path, edit_model + '.json')
            with open(save_path, 'w') as f:
                json.dump(high_level_eval_out, f, indent=4)


        





