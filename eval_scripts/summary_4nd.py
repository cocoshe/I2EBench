import os
import json
import numpy as np

import collections

PROJECT_ROOT = "/home/ma-user/work/mayiwei/yk/new_editbench/EditBench"

ALL_TASKS = [
    # 'Deblurring',       # PURE_SSIM_TASKS  # only used for 1st edit
    # 'HazeRemoval',      # PURE_SSIM_TASKS  # only used for 1st edit
    # 'Lowlight',         # PURE_SSIM_TASKS  # only used for 1st edit
    # 'NoiseRemoval',     # PURE_SSIM_TASKS  # only used for 1st edit
    # 'RainRemoval',      # PURE_SSIM_TASKS  # only used for 1st edit
    # 'ShadowRemoval',    # PURE_SSIM_TASKS  # only used for 1st edit
    # 'SnowRemoval',      # PURE_SSIM_TASKS  # only used for 1st edit
    # 'WatermarkRemoval', # PURE_SSIM_TASKS  # only used for 1st edit
    #################
    'StyleAlteration',  # clip score
    # 'RegionAccuracy',   # special SSIM  # only used for 1st edit
    #################
    'ColorAlteration',
    'Counting',
    #################
    'BGReplacement',       # YES or NO
    'DirectionPerception',
    'ObjectRemoval',
    'Replacement',
]

PURE_SSIM_TASKS = [
    'Deblurring',       # PURE_SSIM_TASKS
    'HazeRemoval',      # PURE_SSIM_TASKS
    'Lowlight',         # PURE_SSIM_TASKS
    'NoiseRemoval',     # PURE_SSIM_TASKS
    'RainRemoval',      # PURE_SSIM_TASKS
    'ShadowRemoval',    # PURE_SSIM_TASKS
    'SnowRemoval',      # PURE_SSIM_TASKS
    'WatermarkRemoval', # PURE_SSIM_TASKS
]

CLIP_SSIM_TASKS = [
    'StyleAlteration',  # clip score
    'RegionAccuracy',   # special SSIM
]

YES_NO_HIGH_LEVEL_TASKS = [
    'BGReplacement',       # others
    'ColorAlteration',   # NUMBER_COLOR_HIGH_LEVEL_TASKS
    'Counting',          # NUMBER_COLOR_HIGH_LEVEL_TASKS
    'DirectionPerception',
    'ObjectRemoval',
    'Replacement',
]

NUMBER_COLOR_HIGH_LEVEL_TASKS = [
    'ColorAlteration',
    'Counting',
]

EDIT_MODELS = [
    'instructpix2pix',
    'magicbrush',
    'mgie',
    'hive',
    'instruct-diffusion',
    'any2pix',
    'iedit',
    'hqedit',
]


for origin_or_diverse in ['ori', '']:
    summary = {}
    suffix = '_ori' if origin_or_diverse == 'ori' else ''
    print(os.path.join(PROJECT_ROOT, 'EditEval' + suffix + "_4nd"))

    for task in ALL_TASKS:
        MAX_SCORE = 0
        MIN_SCORE = 1000
        summary_task = {}
        for model in EDIT_MODELS:
            summary_model = {}
            eval_json_path = os.path.join(PROJECT_ROOT, 'EditEval' + suffix + "_4nd", task, model + '.json')
            with open(eval_json_path, 'r') as f:
                json_data = json.load(f)

            if task in PURE_SSIM_TASKS:
                SSIM_list = []
                for id, eval_info in json_data.items():
                    evaluation = eval_info['evaluation']
                    SSIM = evaluation['SSIM']

                    MAX_SCORE = max(MAX_SCORE, SSIM)
                    MIN_SCORE = min(MIN_SCORE, SSIM)

                    SSIM_list.append(SSIM)
                SSIM_avg = round(float(np.array(SSIM_list).mean()), 4)

                summary_model['SSIM'] = SSIM_avg
                summary_task[model] = summary_model

            elif task == 'StyleAlteration':
                clip_list = []
                for id, eval_info in json_data.items():
                    evaluation = eval_info['evaluation']
                    clip_score = evaluation['clip_score']

                    MAX_SCORE = max(MAX_SCORE, clip_score)
                    MIN_SCORE = min(MIN_SCORE, clip_score)

                    clip_list.append(clip_score)
                clip_avg = round(float(np.array(clip_list).mean()), 4)

                summary_model['clip_score'] = clip_avg
                summary_task[model] = summary_model


            elif task == 'RegionAccuracy':
                region_acc_list = []
                for id, eval_info in json_data.items():
                    evaluation = eval_info['evaluation']
                    region_acc_score = evaluation['RegionAccuracy']

                    MAX_SCORE = max(MAX_SCORE, region_acc_score)
                    MIN_SCORE = min(MIN_SCORE, region_acc_score)

                    region_acc_list.append(region_acc_score)
                region_acc_avg = round(float(np.array(region_acc_list).mean()), 4)

                summary_model['RegionAccuracy'] = region_acc_avg
                summary_task[model] = summary_model
            elif task in NUMBER_COLOR_HIGH_LEVEL_TASKS or \
                 task in YES_NO_HIGH_LEVEL_TASKS:

                samples_num = len(json_data.keys())
                correct_num = 0.
                for id, eval_info in json_data.items():
                    gt = eval_info['gt'].lower()
                    final_judge = eval_info['final_judgement'].lower()

                    if 'yes' in final_judge:
                        correct_num += 1
                acc = correct_num / samples_num
                acc = round(acc, 4)
                metrics = {}
                metrics['accuracy'] = acc
                summary_model['metrics'] = metrics
                summary_task[model] = summary_model

            else:
                raise ValueError('Undifined here!!!')

        if task in PURE_SSIM_TASKS or task in CLIP_SSIM_TASKS:  # get min max value
            summary_task['EXP_MIN'] = MIN_SCORE
            summary_task['EXP_MAX'] = MAX_SCORE
        elif task in YES_NO_HIGH_LEVEL_TASKS or task in NUMBER_COLOR_HIGH_LEVEL_TASKS:
            pass

        summary[task] = collections.OrderedDict(sorted(summary_task.items(), key=lambda t: t[0]))
    
    summary = sorted(summary.items(), key=lambda t: t[0])

    print('summary' + suffix + '_4nd.json')
    with open('./summary' + suffix + '_4nd.json', 'w') as f:
        json.dump(summary, f, indent=4)



