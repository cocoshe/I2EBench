import json
from json import encoder
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
# from metrics_utils.psnr_utils import psnr
from metrics_utils.ssim_utils import ssim

PROJECT_ROOT = "/data/yk/EditBench"
SRC_PATH = "EditData"
DST_PATH = "EditResult"
ORI_DST_PATH = "EditResult_ori"

EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval')
ORI_EVAL_PATH = os.path.join(PROJECT_ROOT, 'EditEval_ori')

if not os.path.exists(EVAL_PATH):
    os.mkdir(EVAL_PATH)
if not os.path.exists(ORI_EVAL_PATH):
    os.mkdir(ORI_EVAL_PATH)

LOW_LEVEL_TASKS = [
    'Deblurring',
    'HazeRemoval',
    'Lowlight',
    'NoiseRemoval',
    'RainRemoval',
    'ShadowRemoval',
    'SnowRemoval',
    'WatermarkRemoval',
    'RegionAccuracy',
]

EDIT_MODELS = [
    # 'hive',
    # 'instructpix2pix',
    # 'magicbrush',
    # 'mgie',
    # 'instruct-diffusion',
    # 'any2pix',
    # 'iedit',
    # 'hqedit',
    'qwen_image_edit',
    'fluxkontext'
]


def find_image_file(base_path):
    """
    查找实际存在的图像文件，支持多种扩展名（png, jpg, jpeg等）
    返回找到的文件路径，如果不存在则返回原始路径
    """
    # 如果文件已经存在，直接返回
    if os.path.exists(base_path):
        return base_path
    
    # 获取不带扩展名的路径和目录
    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # 尝试多种扩展名
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    for ext in extensions:
        candidate_path = os.path.join(base_dir, name_without_ext + ext)
        if os.path.exists(candidate_path):
            return candidate_path
    
    # 如果都没找到，返回原始路径（让后续代码处理错误）
    return base_path


def calc_metrics(task, edit_model, image_name, dst, metric: str, gt: bool = True, mask_name: str = None,): #in_out: str = None):
    gt_or_input = None
    if gt:
        gt_or_input = 'gt'
    else:
        gt_or_input = 'input'

    gt_path = os.path.join(PROJECT_ROOT, SRC_PATH, task, gt_or_input, image_name)
    edited_path = os.path.join(PROJECT_ROOT, dst, task, edit_model, image_name)
    
    # 查找实际存在的图像文件（支持多种扩展名）
    gt_path = find_image_file(gt_path)
    edited_path = find_image_file(edited_path)
    # import pdb; pdb.set_trace()

    gt_img = np.array(Image.open(gt_path))

    mask_img = None
    if mask_name is not None:
        mask_img = np.array(Image.open(os.path.join(PROJECT_ROOT, SRC_PATH, task, 'mask', mask_name)))
    edited_img = np.array(Image.open(edited_path))
    tmp_edited = Image.fromarray(np.uint8(edited_img)).resize(gt_img.shape[:2][::-1])
    edited_img = np.array(tmp_edited)

    if len(gt_img.shape) == 2:
        gt_img = np.expand_dims(gt_img, -1).repeat(3,axis=-1)
    assert gt_img.shape == edited_img.shape, "Shape of gt image: {} doesn't eval to edited image: {}".format(gt_img.shape, edited_img.shape)

    if len(gt_img.shape) == 3:
        gt_img = gt_img.transpose([2, 0, 1]) / 255.0
        edited_img = edited_img.transpose([2, 0, 1]) / 255.0
    elif len(gt_img.shape) == 2:
        print(gt_path)
        gt_img = gt_img / 255.0
        edited_img = edited_img / 255.0
        gt_img = gt_img.reshape([1, gt_img.shape[0], gt_img.shape[1]])
        edited_img = edited_img.reshape([1, edited_img.shape[0], edited_img.shape[1]])
    else:
        raise ValueError('Image shape should be 3 or 1 channel(s), but now get {} channel(s)'.format(len(gt_img.shape)))


    shape = [1] + list(gt_img.shape) # [1,3,H,W]
    gt_img = gt_img.reshape(shape)
    edited_img = edited_img.reshape(shape)
    if metric == 'SSIM':
        if mask_name is not None:
            mask_img = np.expand_dims(mask_img, [0, 1]).repeat(3,axis=1).astype('float')
            cond = mask_img > 0.
            gt_img = np.where(cond, mask_img, gt_img)
            edited_img = np.where(cond, mask_img, edited_img)
        ssim_map = ssim(gt_img, edited_img)
        return ssim_map.mean()
    else:
        raise ValueError('Please choose metric in {}'.format(['SSIM', 'PSNR']))


for dst, val in zip([ORI_DST_PATH,DST_PATH], [ORI_EVAL_PATH,EVAL_PATH]):
    for task_id, task in enumerate(LOW_LEVEL_TASKS):
        path = os.path.join(PROJECT_ROOT, SRC_PATH, task, task + '.json')
        with open(path, 'r') as f:
            data = json.load(f)
        for model_id, edit_model in enumerate(EDIT_MODELS):
            print('Evaluating on [{}] task({}/{}), with [{}] model({}/{}):'.format(task, task_id+1, len(LOW_LEVEL_TASKS), edit_model, model_id+1, len(EDIT_MODELS)))
            low_level_eval_out = {}
            
            if task != 'RegionAccuracy':
                for id, info in tqdm(data.items()):
                    image_name = info['image']
                    dataset = task
                    if dst == ORI_DST_PATH:
                        prompt = info['ori_exp']
                    elif dst == DST_PATH:
                        prompt = info['div_exp']
                    else:
                        raise ValueError('Something wrong here!')
                    evaluation = {}

                    evaluation['SSIM'] = round(float(calc_metrics(task, edit_model, image_name, dst, 'SSIM')), 4)

                    sample = {}
                    sample['image'] = image_name
                    sample['dataset'] = dataset
                    sample['prompt'] = prompt
                    sample['evaluation'] = evaluation
                    sample['type'] = info['type']

                    low_level_eval_out[id] = sample

            elif task == 'RegionAccuracy':
                in_scores = []
                out_scores = []
                region_accs = []
                for id, info in tqdm(data.items()):
                    image_name = info['image']
                    mask_name = info['mask']
                    dataset = task
                    if dst == ORI_DST_PATH:
                        prompt = info['ori_exp']
                    elif dst == DST_PATH:
                        prompt = info['div_exp']
                    else:
                        raise ValueError('Something wrong here!')
                    evaluation = {}


                    edited_path = os.path.join(PROJECT_ROOT, dst, task, edit_model, image_name)
                    # 查找实际存在的图像文件（支持多种扩展名）
                    edited_path = find_image_file(edited_path)
                    mask_path = os.path.join(PROJECT_ROOT, SRC_PATH, task, 'mask', mask_name)
                    
                    ssim_score = calc_metrics(task, edit_model, image_name, dst, 'SSIM', gt=False, mask_name=mask_name)

                    print('-'*30)
                    print(f'task: {task}, edit_model: {edit_model}, ssim_score: {ssim_score}')
                    print('-'*30)

                    evaluation['RegionAccuracy'] = round(float(ssim_score), 4)

                    sample = {}
                    sample['image'] = image_name
                    sample['dataset'] = dataset
                    sample['prompt'] = prompt
                    sample['evaluation'] = evaluation
                    sample['type'] = info['type']

                    low_level_eval_out[id] = sample
            else:
                raise ValueError('Wrong task!')


            
            task_path = os.path.join(val, task)
            if not os.path.exists(task_path):
                os.mkdir(task_path)

            save_path = os.path.join(task_path, edit_model + '.json')
            with open(save_path, 'w') as f:
                json.dump(low_level_eval_out, f, indent=4)


        





