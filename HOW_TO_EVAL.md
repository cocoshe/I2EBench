# HOW TO EVAL

## Edit the images with your own editing model

1. Download the dataset from [Google Drive](https://img.shields.io/badge/Dataset-Download-green?logo=googlechrome&logoColor=green)

2. Write a script to edit the images from the dataset, the image meta information(image names, image original expression, diverse expression...) can be get from the json, e.g. `EditData/BGReplacement.json`

   ```json
   {
       "1": {
           "image": "0001.jpg",
           "ori_exp": "Change the background of this photo to snow",
           "Evaluation": "Is the background of this image snow? Answer Yes or No.",
           "Answer": "Yes",
           "type": "Scenery",
           "div_exp": "Transform this image into a snowy scene"
       },
       "2": {
           "image": "0002.jpg",
           "ori_exp": "Change the background of this photo to snow",
           "Evaluation": "Is the background of this image meadow? Answer Yes or No.",
           "Answer": "No",
           "type": "Scenery",
           "div_exp": "Alter the backdrop of this picture to depict snow"
       },
   ...
   ```

   and the edited result should be saved in `EditResult_ori` for images edited with `ori_exp`, `EditResult` for images edited with `div_exp`, e. g. `EditResult/BGReplacement/{Your Model Name}/`

   The **edited image names** should be the same with the **original image names**, e.g. `0001.jpg`, `0002.jpg`.

## Eval with the low-level metrics

**Check the `low_level_eval.py` script in `eval_scripts`**

1. Set your absolute path to the `PROJECT_ROOT` in the script, e.g. `/home/EditBench`

2. Modify the `EDIT_MODELS`, add `{Your Model Name}`, which must be the same name as the model name in the `EditResult` folder

   ```python
   EDIT_MODELS = [
   #    'hive',
   #    'instructpix2pix',
   #    'magicbrush',
   #    'mgie',
   #    'instruct-diffusion',
   #    'any2pix',
   #    'iedit',
   #    'hqedit',
       '{Your Model Name}',
   ]
   ```

3. Run the script

## Eval with the high-level metrics

+ Stage1: `high_level_eval_stage1.py`, the purpose of this stage is **getting the answers in edited images**

  1. Set your absolute path to the `PROJECT_ROOT` in the script, e.g. `/home/EditBench`

  2. Modify the `EDIT_MODELS`, add `{Your Model Name}`, which must be the same name as the model name in the `EditResult` folder

     ```python
     EDIT_MODELS = [
     #    'hive',
     #    'instructpix2pix',
     #    'magicbrush',
     #    'mgie',
     #    'instruct-diffusion',
     #    'any2pix',
     #    'iedit',
     #    'hqedit',
         '{Your Model Name}',
     ]
     ```

  3. Set your own `GPT4V` configuration in `eval_scripts/metrics_utils/gpt4v_utils.py`

  4. Run the script

+ Stage2: `high_level_eval_stage2_final_judge.py`, the purpose of this stage is **getting the more robust answers**

  1. Set your absolute path to the `PROJECT_ROOT` in the script, e.g. `/home/EditBench`

  2. Modify the `EDIT_MODELS`, add `{Your Model Name}`, which must be the same name as the model name in the `EditResult` folder

     ```python
     EDIT_MODELS = [
     #    'hive',
     #    'instructpix2pix',
     #    'magicbrush',
     #    'mgie',
     #    'instruct-diffusion',
     #    'any2pix',
     #    'iedit',
     #    'hqedit',
         '{Your Model Name}',
     ]
     ```

  3. Set your own `ChatGPT` configuration in `eval_scripts/metrics_utils/gpt4v_utils.py`

  4. Run the script

## Summarization

**Check the `summary.py` script in `eval_scripts`**

1. Set your absolute path to the `PROJECT_ROOT` in the script, e.g. `/home/EditBench`

2. Modify the `EDIT_MODELS`, add `{Your Model Name}`, which must be the same name as the model name in the `EditResult` folder

   ```python
   EDIT_MODELS = [
   #    'hive',
   #    'instructpix2pix',
   #    'magicbrush',
   #    'mgie',
   #    'instruct-diffusion',
   #    'any2pix',
   #    'iedit',
   #    'hqedit',
       '{Your Model Name}',
   ]
   ```

3. Run the script

**The example output of the script can be found in `summary.json`**

## Summarization with the types in each model

**Check the `summary_model_type_avg_score.py` script in `eval_scripts`**

1. Set your absolute path to the `PROJECT_ROOT` in the script, e.g. `/home/EditBench`

2. Modify the `EDIT_MODELS`, add `{Your Model Name}`, which must be the same name as the model name in the `EditResult` folder

   ```python
   EDIT_MODELS = [
   #    'hive',
   #    'instructpix2pix',
   #    'magicbrush',
   #    'mgie',
   #    'instruct-diffusion',
   #    'any2pix',
   #    'iedit',
   #    'hqedit',
       '{Your Model Name}',
   ]
   ```

3. Run the script

**The example output of the script can be found in `summary_model_type_avg_score.json`**

