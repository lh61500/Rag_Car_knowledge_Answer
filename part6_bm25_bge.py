import json

import pdfplumber

model=json.load(open('submit_embedding_top10.json',encoding='utf8'))
bm25=json.load(open('submit_bm25_retriecal_top10.json',encoding='utf8'))
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

fusion_result=[]
#以页码为基础进行排序 页码越小权重越低 也好像不太合理
k=60#文献验证
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
    fusion_result.append(q1)


with open('submit_fusion_model+bm25_retrieval.json','w',encoding='utf8') as up:
    json.dump(fusion_result,up,ensure_ascii=False,indent=4)