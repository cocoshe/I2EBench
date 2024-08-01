import json
from json import encoder
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from metrics_utils.psnr_utils import psnr
from metrics_utils.ssim_utils import ssim

PROJECT_ROOT = "/path/to/project"
SRC_PATH = "EditData"
DST_PATH = "EditResult"
ORI_DST_PATH = "EditResult_ori"

RANK_PATH = 'EditRank'
ORI_RANK_PATH = 'EditRank_ori'

EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval')
ORI_EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_ori')

if not os.path.exists(EVAL_PATH):
    os.mkdir(EVAL_PATH)
if not os.path.exists(ORI_EVAL_PATH):
    os.mkdir(ORI_EVAL_PATH)

SCORE_BASED_TASKS = [
    'Deblurring',
    'HazeRemoval',
    'Lowlight',
    'NoiseRemoval',
    'RainRemoval',
    'ShadowRemoval',
    'SnowRemoval',
    'WatermarkRemoval',
    'RegionAccuracy',    # SSIM score
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

def run_score_based_task_rank():
    for rank, val in zip([ORI_RANK_PATH,RANK_PATH], [ORI_EVAL_PATH,EVAL_PATH]):
        for task_id, task in enumerate(SCORE_BASED_TASKS):
            print(f'task: {task}[{task_id+1}/{len(SCORE_BASED_TASKS)}]')
            path = os.path.join(PROJECT_ROOT, SRC_PATH, task, task + '.json')
            with open(path, 'r') as f:
                data = json.load(f)

            name_rank_dict = {}
            for id, info in data.items():
                img_name = info['image']
                model_score_list = []
                eval_info = None
                for model_id, edit_model in enumerate(EDIT_MODELS):
                    eval_json_path = os.path.join(PROJECT_ROOT, val, task, edit_model + '.json')
                    with open(eval_json_path, 'r') as f:
                        eval_json = json.load(f)
                    flag = False
                    for eval_id, tmp_eval_info in eval_json.items():
                        eval_img_name = tmp_eval_info['image']
                        if eval_img_name == img_name:
                            eval_info = tmp_eval_info
                            eval_dataset = tmp_eval_info['dataset']
                            assert eval_dataset == task, f'eval dataset `{eval_dataset}` is not equal to task name `{task}`'
                            flag = True
                            evaluation = tmp_eval_info['evaluation']
                            score = list(evaluation.values())[0]  # SSIM or RegionAccuracy or clip_score
                            model_score_list.append([edit_model, score])
                            break
                    if not flag:
                        raise ValueError(f'Fail to find image name `{img_name}` in eval json')
                model_score_list.sort(key=lambda x: x[1], reverse=True)
                

                name_rank_dict[id] = eval_info
                name_rank_dict[id]['evaluation'] = list(name_rank_dict[id]['evaluation'].keys())[0]
                name_rank_dict[id]['model_rank'] = model_score_list

            rank_dir = os.path.join(PROJECT_ROOT, rank)
            if not os.path.exists(rank_dir):
                os.mkdir(rank_dir)

            eval_save_path = os.path.join(PROJECT_ROOT, rank, task + '.json')
            print(f'save path: {eval_save_path}')
            with open(eval_save_path, 'w') as f:
                json.dump(name_rank_dict, f, indent=4)

        

def run_right_wrong_task_rank():
    for rank, val in zip([ORI_RANK_PATH,RANK_PATH], [ORI_EVAL_PATH,EVAL_PATH]):
        for task_id, task in enumerate(RIGHT_WRONG_TASKS):
            print(f'task: {task}[{task_id+1}/{len(RIGHT_WRONG_TASKS)}]')
            path = os.path.join(PROJECT_ROOT, SRC_PATH, task, task + '.json')
            with open(path, 'r') as f:
                data = json.load(f)

            name_rank_dict = {}
            for id, info in data.items():
                img_name = info['image']
                right_model_list, wrong_model_list = [], []

                eval_info = None
                for model_id, edit_model in enumerate(EDIT_MODELS):
                    eval_json_path = os.path.join(PROJECT_ROOT, val, task, edit_model + '.json')
                    with open(eval_json_path, 'r') as f:
                        eval_json = json.load(f)
                    flag = False
                    for eval_id, tmp_eval_info in eval_json.items():
                        eval_img_name = tmp_eval_info['image']
                        if eval_img_name == img_name:
                            eval_info = tmp_eval_info
                            eval_dataset = tmp_eval_info['dataset']
                            assert eval_dataset == task, f'eval dataset `{eval_dataset}` is not equal to task name `{task}`'
                            flag = True
                            final_judgement = tmp_eval_info['final_judgement']
                            if 'yes' in final_judgement.lower():
                                right_model_list.append(edit_model)
                            else:
                                wrong_model_list.append(edit_model)
                            break
                    if not flag:
                        raise ValueError(f'Fail to find image name `{img_name}` in eval json')
                

                name_rank_dict[id] = eval_info
                name_rank_dict[id]['evaluation'] = list(name_rank_dict[id]['evaluation'].keys())[0]
                name_rank_dict[id]['model_rank'] = {
                    'right_models': right_model_list,
                    'wrong_models': wrong_model_list,
                }

            rank_dir = os.path.join(PROJECT_ROOT, rank)
            if not os.path.exists(rank_dir):
                os.mkdir(rank_dir)

            eval_save_path = os.path.join(PROJECT_ROOT, rank, task + '.json')
            print(f'save path: {eval_save_path}')
            with open(eval_save_path, 'w') as f:
                json.dump(name_rank_dict, f, indent=4)


run_score_based_task_rank()
run_right_wrong_task_rank()
