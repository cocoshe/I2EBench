import json
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data_path = '/home/ma-user/work/mayiwei/yk/new_editbench/EditBench/EditData'
tasks = os.listdir(data_path)

jsons = []
for task in tasks:
    for f in os.listdir(os.path.join(data_path, task)):
        if f.endswith('json'):
            jsons.append(os.path.join(data_path, task, f))

oris, divs = [], []
for j in jsons:
    with open(j, 'r') as f:
        d = json.load(f)
        for k, v in d.items():
            oris.append(v['ori_exp'])
            divs.append(v['div_exp'])

# oris.remove('image')
# oris.remove('picture')
# oris.remove('photo')

# divs.remove('image')
# divs.remove('picture')
# divs.remove('photo')


# 将所有文本合并为一个字符串
all_ori = " ".join(oris)
all_div = " ".join(divs)

all_ori = all_ori.replace('image', '').replace('picture', '').replace('photo', '')
all_div = all_div.replace('image', '').replace('picture', '').replace('photo', '')


# 生成词云
wordcloud = WordCloud(
    width=997, height=816,
    background_color='white',
    max_words=200,
    colormap='viridis'
).generate(all_ori)

wordcloud.to_file('./ori_wordcloud.png')

# # 显示词云
# plt.figure(figsize=(9.97, 8.16))
# # plt.figure()
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
# plt.savefig('./ori_wordcloud.jpg')

# 生成词云
wordcloud = WordCloud(
    width=997, height=816,
    background_color='white',
    max_words=200,
    colormap='viridis'
).generate(all_div)

wordcloud.to_file('./div_wordcloud.png')

# plt.figure(figsize=(9.97, 8.16))
# # plt.figure()
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
# plt.savefig('./div_wordcloud.jpg')

print(oris)
print(len(oris))
print('-'*100)
print(divs)
print(len(divs))