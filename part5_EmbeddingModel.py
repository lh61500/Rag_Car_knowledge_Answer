import json

import pdfplumber
from sentence_transformers import SentenceTransformer
##基于语义的编码
#读取数据集
questions=json.load(open("questions.json",encoding='utf8'))
pdf=pdfplumber.open("初赛训练数据集.pdf")
pdf_content=[]

#文档切分
def split_text_fixed_size(text,chunk_size):
    return  [text[i:i+chunk_size]for i in range(0,len(text),chunk_size)]
#分块之后的文本还是来自于这一页
for page_idx in range(len(pdf.pages)):
    text=pdf.pages[page_idx].extract_text()
    for chunk_text in split_text_fixed_size(text,40):
        pdf_content.append(
            {
                'page':'page_'+str(page_idx+1),
                'content':chunk_text
            }
        )

model = SentenceTransformer("infgrad/stella-large-zh-v3-1792d")
question_sentences =[x['question' ]for x in questions]
pdf_content_sentences=[x['content'] for x in pdf_content]

question_embeddings= model.encode(question_sentences,normalize_embeddings=True)
pdf_embeddings=model.encode(pdf_content_sentences,normalize_embeddings=True)
def remove_dulicates(input_list):
    seen=set()
    result=[]
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result



for query_idx,feat in enumerate(question_embeddings):
    score=feat@pdf_embeddings.T
    max_score_page_idx=score.argsort()[::-1]
    pages=[pdf_content[x]['page'] for x in max_score_page_idx]
    questions[query_idx]['reference']=remove_dulicates(pages[:100][:10])

with open('submit_embedding_top10.json','w',encoding='utf8') as up:
    json.dump(questions,up,ensure_ascii=False,indent=4)

#BGE

