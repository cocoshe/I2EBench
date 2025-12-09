import json
from json import encoder
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from metrics_utils.ssim_utils import ssim
from collections import defaultdict
from copy import deepcopy

PROJECT_ROOT = "/home/myw/yiwei/code/EditBench"
# SRC_PATH = "EditData"
# DST_PATH = "EditResult"
# ORI_DST_PATH = "EditResult_ori"

RANK_PATH = 'EditRank_4nd'
ORI_RANK_PATH = 'EditRank_ori_4nd'

# EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval')
# ORI_EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_ori')
EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_4nd')
ORI_EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_ori_4nd')

if not os.path.exists(EVAL_PATH):
    os.mkdir(EVAL_PATH)
if not os.path.exists(ORI_EVAL_PATH):
    os.mkdir(ORI_EVAL_PATH)

SCORE_BASED_TASKS = [
    # 'Deblurring',  # Only used in 1st edit
    # 'HazeRemoval',
    # 'Lowlight',
    # 'NoiseRemoval',
    # 'RainRemoval',
    # 'ShadowRemoval',
    # 'SnowRemoval',
    # 'WatermarkRemoval',
    # 'RegionAccuracy',    # SSIM score
] + ['StyleAlteration']  # add clip_score rank

RIGHT_WRONG_TASKS = [
    'BGReplacement',
    'Counting',
    'DirectionPerception',
    'Replacement',
    'ObjectRemoval',
    'ColorAlteration',
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

all_types = {
    'animal',
    'object',
    'scenery',
    'plant',
    'human',
    'global',
}

SSIM_BASED_TASK = [
    'Deblurring',
    'HazeRemoval',
    'Lowlight',
    'NoiseRemoval',
    'RainRemoval',
    'ShadowRemoval',
    'SnowRemoval',
    'WatermarkRemoval',
    'RegionAccuracy',    # SSIM score
]

CLIP_MAX = 35.0
CLIP_BASED_TASK = [
    'StyleAlteration'
]


def calc_model_type_avg_score():
    ori_div_dict = {}
    for _, val in zip([ORI_RANK_PATH,RANK_PATH], [ORI_EVAL_PATH,EVAL_PATH]):
        model_type_dict = {}
        for model_id, edit_model in enumerate(EDIT_MODELS):
            type_sum_score_dict = {
                'eval_metrics': {
                    'ssim': {
                        'sum': 0,
                        'num': 0,
                    },
                    'uniformed_clip': {
                        'sum': 0,
                        'num': 0,
                    },
                    'vlm': {
                        'sum': 0,
                        'num': 0,
                    },
                },
                'score': 0,
            }
            model_type_dict[edit_model] = {}
            for _type in all_types:
                model_type_dict[edit_model].update({
                    _type: deepcopy(type_sum_score_dict)
                })
        for model_id, edit_model in enumerate(EDIT_MODELS):
            for task_id, task in enumerate(SCORE_BASED_TASKS + RIGHT_WRONG_TASKS):
                print(f'task: {task}[{task_id+1}/{len(SCORE_BASED_TASKS + RIGHT_WRONG_TASKS)}]')
                path = os.path.join(PROJECT_ROOT, val, task, edit_model + '.json')
                with open(path, 'r') as f:
                    data = json.load(f)

                name_rank_dict = {}
                for id, info in data.items():
                    img_name = info['image']
                    img_type = None
                    if task == 'StyleAlteration':
                        img_type = 'Global'
                    else:
                        img_type = info['type']
                    print('edit_model: ', edit_model, " img_type: ", img_type, ' id: ', id, ' task: ', task)
                    if img_type not in model_type_dict[edit_model].keys():
                        img_type = img_type.lower()

                    eval_score = None
                    if task in SCORE_BASED_TASKS:
                        eval_score = list(info['evaluation'].values())[0]
                    elif task in RIGHT_WRONG_TASKS:
                        final = info['final_judgement']
                        if 'yes' in final.lower():
                            eval_score = 1
                        else:
                            eval_score = 0
                    else:
                        assert "Wrong task !!!!"

                    if task in SSIM_BASED_TASK:
                        model_type_dict[edit_model][img_type]['eval_metrics']['ssim']['num'] += 1
                        model_type_dict[edit_model][img_type]['eval_metrics']['ssim']['sum'] += eval_score
                    elif task in CLIP_BASED_TASK:
                        model_type_dict[edit_model][img_type]['eval_metrics']['uniformed_clip']['num'] += 1
                        model_type_dict[edit_model][img_type]['eval_metrics']['uniformed_clip']['sum'] += eval_score / CLIP_MAX
                    elif task in RIGHT_WRONG_TASKS:
                        model_type_dict[edit_model][img_type]['eval_metrics']['vlm']['num'] += 1
                        model_type_dict[edit_model][img_type]['eval_metrics']['vlm']['sum'] += eval_score


            for _type in all_types:
                score1 = model_type_dict[edit_model][_type]['eval_metrics']['ssim']['sum'] / model_type_dict[edit_model][_type]['eval_metrics']['ssim']['num'] if model_type_dict[edit_model][_type]['eval_metrics']['ssim']['num'] != 0 else 0
                score2 = model_type_dict[edit_model][_type]['eval_metrics']['uniformed_clip']['sum'] / model_type_dict[edit_model][_type]['eval_metrics']['uniformed_clip']['num'] if model_type_dict[edit_model][_type]['eval_metrics']['uniformed_clip']['num'] != 0 else 0
                score3 = model_type_dict[edit_model][_type]['eval_metrics']['vlm']['sum'] / model_type_dict[edit_model][_type]['eval_metrics']['vlm']['num'] if model_type_dict[edit_model][_type]['eval_metrics']['vlm']['num'] != 0 else 0
                model_type_dict[edit_model][_type]['score'] = round((score1 + score2 + score3) / 3, 4)
        ori_div_dict[val.split('/')[-1]] = model_type_dict
    print(ori_div_dict)
    
    with open('summary_model_type_avg_score_4nd.json', 'w') as f:
        json.dump(ori_div_dict, f, indent=4)


calc_model_type_avg_score()

