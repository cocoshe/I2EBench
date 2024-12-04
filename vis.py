import streamlit as st
import os
from PIL import Image
import json

# 定义目录
edit_results_dirs = ['EditResult_ori', 'EditResult_ori_2nd', 'EditResult_ori_3nd', 'EditResult_ori_4nd', 'EditResult_ori_5nd']

# edit_results_dirs = ['EditResult', 'EditResult_2nd', 'EditResult_3nd', 'EditResult_4nd', 'EditResult_5nd']

tasks_dir = ['BGReplacement', 'ColorAlteration', 'Counting', 'DirectionPerception', 'ObjectRemoval', 'Replacement', 'StyleAlteration']
models_dir = ['any2pix', 'hive', 'hqedit', 'iedit', 'instruct-diffusion', 'instructpix2pix', 'magicbrush', 'mgie']
edit_prompt = ['', '_2nd', '_3nd', '_4nd', '_5nd']

resize_shape = (224, 224)
# 加载并显示图片的函数
def display_images(task):
    image_names = os.listdir(os.path.join('EditResult_ori_5nd', task, 'any2pix'))
    # image_names = os.listdir(os.path.join('EditResult_5nd', task, 'any2pix'))

    # resize_temp = {}
    for image_name in image_names:

        # 创建多个列，以便将图片水平排列
        cols = st.columns(len(models_dir))  # 为每个模型创建一个列

        for i in range(len(models_dir)):
            with cols[i]:
                ori = Image.open(os.path.join('EditData', task, 'input', image_name))
                # st.image(ori, caption=f"{model}_ori: {image_name}", use_container_width=True)
                st.image(ori, caption=f"origin", use_container_width=True)

        # 遍历每个编辑结果目录（例如 EditResult_ori, EditResult_ori_2nd 等）
        for res_id, edit_result in enumerate(edit_results_dirs):
            json_p = os.path.join('EditData', task, task + edit_prompt[res_id] + '.json')
            with open(json_p, 'r') as f:
                prompts = json.load(f)
            prompt = None
            for id in prompts.keys():
                if image_name == prompts[id]['image']:
                    prompt = prompts[id]['ori_exp']
                    break
            st.subheader(f"Results from {edit_result}, \nprompt: {prompt}")
          

            # 创建多个列，以便将图片水平排列
            cols = st.columns(len(models_dir))  # 为每个模型创建一个列

            
            # ori = Image.open(os.path.join('EditData', task, 'input', image_name))
            # # st.image(ori, caption=f"{model}_ori: {image_name}", use_container_width=True)
            # st.image(ori, caption=f"origin", use_container_width=True)

            # 遍历每个模型，并在对应的列中显示图片
            for i, model in enumerate(models_dir):
                model_images = []  # 存储每个模型的图片
                
                # 构建当前编辑结果和任务目录的路径
                model_task_dir = os.path.join(edit_result, task, model)
                if os.path.exists(model_task_dir):
                    image_path = os.path.join(model_task_dir, image_name)
                    if os.path.exists(image_path):
                        img = Image.open(image_path).resize(resize_shape)
                        # if resize_temp.get(edit_result, None) is None:
                        #     resize_temp[edit_result] = img.size
                        # else:
                        #     img = img.resize(resize_temp[edit_result])
                        model_images.append((image_name, img))

                # 在对应的列中显示当前模型的图片
                if model_images:
                    with cols[i]:
                        # ori = Image.open(os.path.join('EditData', task, 'input', image_name))
                        # # st.image(ori, caption=f"{model}_ori: {image_name}", use_container_width=True)
                        # st.image(ori, caption=f"origin", use_container_width=True)
                        for image_name, img in model_images:
                            st.image(img, caption=f"{model}_{res_id+1}r: {image_name}", use_container_width=True)

# Streamlit 主函数
def main():
    st.title("Model Image Comparison")
    task = st.selectbox('Select Task', tasks_dir)  # 用户可以选择任务

    if task:
        display_images(task)

if __name__ == "__main__":
    main()
