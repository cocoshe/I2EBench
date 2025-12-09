import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import json

# 定义目录
edit_results_dirs = ['EditResult_ori', 'EditResult_ori_2nd', 'EditResult_ori_3nd', 'EditResult_ori_4nd', 'EditResult_ori_5nd']

# edit_results_dirs = ['EditResult', 'EditResult_2nd', 'EditResult_3nd', 'EditResult_4nd', 'EditResult_5nd']

tasks_dir = ['BGReplacement', 'ColorAlteration', 'Counting', 'DirectionPerception', 'ObjectRemoval', 'Replacement', 'StyleAlteration']
models_dir = ['any2pix', 'hive', 'hqedit', 'iedit', 'instruct-diffusion', 'instructpix2pix', 'magicbrush', 'mgie', 'fluxkontext', 'qwen_image_edit']
edit_prompt = ['', '_2nd', '_3nd', '_4nd', '_5nd']


task_dir_cast = {
    'BGReplacement': 'Background Replacement',
    'ColorAlteration': 'Color Alteration',
    'Counting': 'Counting',
    'DirectionPerception': 'Direction Perception',
    'ObjectRemoval': 'Object Removal',
    'Replacement': 'Replacement',
    'StyleAlteration': 'Style Alteration',
}

resize_shape = (224, 224)

# 保存对比图像的函数
def save_comparison_image(task, image_name, origin_size):
    """
    保存排版好的对比图像
    包含6行：origin, 1nd, 2nd, 3nd, 4nd, 5nd
    每行包含所有模型的图像
    """
    num_models = len(models_dir)
    num_rows = 6  # origin + 5个编辑结果
    
    # 计算每张图像的尺寸（使用origin的尺寸）
    img_width, img_height = origin_size
    
    # 创建大图像：每行有num_models个图像，共num_rows行
    # 添加一些间距
    spacing = 10  # 列间距
    # row_spacing = 30  # 行间距（增加行之间的距离）
    row_spacing = 180  # 行间距（增加行之间的距离）
    total_width = num_models * img_width + (num_models + 1) * spacing
    total_height = num_rows * img_height + (num_rows - 1) * row_spacing + 2 * spacing + 30  # 额外空间用于标签
    
    # 创建白色背景
    result_image = Image.new('RGB', (total_width, total_height), color='white')
    
    # 绘制第一行：origin图像（每个模型位置都显示origin）
    row = 0
    origin_img = Image.open(os.path.join('EditData', task, 'input', image_name))
    origin_img = origin_img.resize(origin_size)
    
    for col in range(num_models):
        x = spacing + col * (img_width + spacing)
        y = spacing + 30 + row * (img_height + row_spacing)
        result_image.paste(origin_img, (x, y))
    
    # 绘制标签
    draw = ImageDraw.Draw(result_image)
    font = ImageFont.load_default()
    
    # # 绘制行标签
    # row_labels = ['Origin', '1nd', '2nd', '3nd', '4nd', '5nd']
    # for i, label in enumerate(row_labels):
    #     y = spacing + 30 + i * (img_height + spacing) + img_height // 2
    #     draw.text((5, y), label, fill='black', font=font, anchor='lm')
    
    # # 绘制列标签（模型名称）
    # for col, model in enumerate(models_dir):
    #     x = spacing + col * (img_width + spacing) + img_width // 2
    #     y = 15
    #     # 只显示模型名称的前几个字符，避免太长
    #     model_short = model[:15] if len(model) > 15 else model
    #     draw.text((x, y), model_short, fill='black', font=font, anchor='mt')
    
    # 绘制后续行：1nd到5nd的编辑结果
    for res_id, edit_result in enumerate(edit_results_dirs):
        row = res_id + 1
        for col, model in enumerate(models_dir):
            model_task_dir = os.path.join(edit_result, task, model)
            image_path = os.path.join(model_task_dir, image_name)
            
            if os.path.exists(image_path):
                img = Image.open(image_path).resize(origin_size)
            else:
                # 如果图像不存在，创建黑色占位符
                img = Image.new('RGB', origin_size, color='black')
            
            x = spacing + col * (img_width + spacing)
            y = spacing + 30 + row * (img_height + row_spacing)
            result_image.paste(img, (x, y))
    
    # 保存图像
    save_dir = os.path.join('saved_comparisons', task)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, image_name)
    result_image.save(save_path)
    
    return save_path

# 加载并显示图片的函数
def display_images(task):
    image_names = os.listdir(os.path.join('EditResult_ori_5nd', task, 'any2pix'))
    # image_names = os.listdir(os.path.join('EditResult_5nd', task, 'any2pix'))

    # resize_temp = {}
    for image_name in image_names:
        # 加载并记录origin图像的尺寸
        origin_path = os.path.join('EditData', task, 'input', image_name)
        origin_img = Image.open(origin_path)
        origin_size = origin_img.size  # 记录origin图像的尺寸
        
        # 添加保存按钮
        if st.button(f"保存对比图: {image_name}", key=f"save_{image_name}"):
            save_path = save_comparison_image(task, image_name, origin_size)
            st.success(f"图像已保存到: {save_path}")

        # 创建多个列，以便将图片水平排列
        cols = st.columns(len(models_dir))  # 为每个模型创建一个列

        for i in range(len(models_dir)):
            with cols[i]:
                # st.image(ori, caption=f"{model}_ori: {image_name}", use_container_width=True)
                # st.image(ori, caption=f"origin", use_container_width=True)
                st.image(origin_img, caption=f"origin", use_column_width=True)

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
            st.subheader(f"Results from {edit_result}, \n{task_dir_cast[task]}: {prompt}")
          

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
                        # 使用origin的尺寸进行resize
                        img = Image.open(image_path).resize(origin_size)
                        model_images.append((image_name, img))

                # 在对应的列中显示当前模型的图片
                if model_images:
                    with cols[i]:
                        # ori = Image.open(os.path.join('EditData', task, 'input', image_name))
                        # # st.image(ori, caption=f"{model}_ori: {image_name}", use_container_width=True)
                        # st.image(ori, caption=f"origin", use_container_width=True)
                        for image_name, img in model_images:
                            # st.image(img, caption=f"{model}_{res_id+1}r: {image_name}", use_container_width=True)
                            st.image(img, caption=f"{model}_{res_id+1}r: {image_name}", use_column_width=True)

# Streamlit 主函数
def main():
    st.title("Model Image Comparison")
    task = st.selectbox('Select Task', tasks_dir)  # 用户可以选择任务

    if task:
        display_images(task)

if __name__ == "__main__":
    main()
