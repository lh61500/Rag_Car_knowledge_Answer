import json
import jieba
import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
#读取数据集
questions=json.load(open("questions.json",encoding='utf8'))
pdf=pdfplumber.open("初赛训练数据集.pdf")
pdf_content=[]
for page_idx in range(len(pdf.pages)):
    pdf_content.append(
        {
            'page':'page_'+str(page_idx+1),
            'content':pdf.pages[page_idx].extract_text()
        }
    )
#对提问和pdf内容进行分词
question_words=[''.join(jieba.lcut(x['question'])) for x in questions]
pdf_content_words=[''.join(jieba.lcut(x['content'])) for x in pdf_content]

tfidf=TfidfVectorizer()
tfidf.fit(question_words+pdf_content_words)

#提取TFIDF
question_feat=tfidf.transform(question_words)
pdf_content_feat=tfidf.transform(pdf_content_words)

#进行归一化
question_feat=normalize(question_feat)
pdf_content_feat=normalize(pdf_content_feat)

#检索进行排序
for query_idx,feat in enumerate(question_feat):
    score=feat @ pdf_content_feat.T
    score=score.toarray()[0]
    max_score_page_idx=score.argsort()[-1]+1
    questions[query_idx]['reference']='page_'+str(max_score_page_idx)

with open('submit.json','w',encoding='utf8') as up:
    json.dump(questions,up,ensure_ascii=False,indent=4)

