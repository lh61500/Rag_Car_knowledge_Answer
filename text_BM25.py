import json

import jieba
import pdfplumber

questions=json.load(open("questions.json",encoding='utf8'))
pdf= pdfplumber.open("初赛训练数据集.pdf")
pdf_content=[]
for page_idx in range(len(pdf.pages)):
    pdf_content.append(
        {
            'page':'page_'+str(page_idx+1),
            'content':pdf.pages[page_idx].extract_text()
        }
    )

from  rank_bm25 import BM25Okapi

pdf_content_words=[jieba.lcut(x['content']) for x in pdf_content]
bm25=BM25Okapi(pdf_content_words)

for query_idx in range(len(questions)):
    doc_scores=bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
    max_score_page_idx=doc_scores.argsort()[::-1]+1
    questions[query_idx]['reference']=['page_'+str(x) for x in max_score_page_idx[:10]]

with open('submit_bm25_retriecal_top10.json','w',encoding='utf8') as up:
    json.dump(questions,up,ensure_ascii=False,indent=4)

