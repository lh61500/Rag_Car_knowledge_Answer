import json

import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 读取数据集
questions = json.load(open("questions.json", encoding='utf-8'))
pdf = pdfplumber.open("初赛训练数据集.pdf")
pdf_content = []

for page_idx in range(len(pdf.pages)):
    pdf_content.append(
        {
            'page': 'page_' + str(page_idx + 1),
            'content': pdf.pages[page_idx].extract_text()
        }
    )
# 加载排序模型 文本重排序模型
# 重排序模型选取infgrad/stella-mrl-large-zh-v3.5-1792d
tokenizer = AutoTokenizer.from_pretrained('infgrad/stella-mrl-large-zh-v3.5-1792d')
rerank_model = AutoModelForSequenceClassification.from_pretrained('infgrad/stella-mrl-large-zh-v3.5-1792d')
rerank_model.cuda()
rerank_model.eval()

# 进行召回合并
model = json.load(open('submit_embedding_top10.json', encoding='utf8'))
bm25 = json.load(open('submit_bm25_retriecal_top10.json', encoding='utf8'))

fusion_result = []
# 以页码为基础进行排序 页码越小权重越低 也好像不太合理
k = 60  # 文献验证

for q1,q2 in zip(bm25,model):
    fusion_score={}
    for idx,q in enumerate(q1['reference']):
        if q not in fusion_score:
            fusion_score[q]=1/(idx+k)
        else:
            fusion_score[q] +=1/(idx+k)

    for idx,q in enumerate(q2['reference']):
        if q not in fusion_score:
            fusion_score[q]=1/(idx+k)
        else:
            fusion_score[q] +=1/(idx+k)
    sorted_dict= sorted(fusion_score.items(),key=lambda item:item[1],reverse=True)
    q1['reference']=sorted_dict[0][0]
print(sorted_dict)
# 这里需要进行配对 将文本对用户提问与检索得到结果进行拼接
pairs = []
for sorted_result in sorted_dict[:3]:
    page_id = sorted_result[0]  # 获取页码
    page_index = int(page_id.split('_')[1]) - 1  # 将页码转换为索引
    content=pdf_content[page_index]['content']
    pairs.append([questions[0]['question'],content])
print(pairs)
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
with torch.no_grad():
    inputs = {key: inputs[key].cuda() for key in inputs.keys()}
    scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()  # 是否匹配进行打分
fusion_result = []
sorted_result = sorted_dict[scores.cpu().numpy().argmax()]  # 重排序模型排序结果
print(sorted_result)
for num in sorted_result:
 q1['reference'] = sorted_result[0]
 fusion_result.append(q1)
with open('submit_part6_fusion_rerank.json', 'w', encoding='utf8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)