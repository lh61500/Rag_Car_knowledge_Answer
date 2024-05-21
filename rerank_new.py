import jieba, json, pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi

questions = json.load(open("questions.json",encoding='utf-8'))

pdf = pdfplumber.open("初赛训练数据集.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })

# 加载重排序模型
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('aspire/acge_text_embedding')
rerank_model = AutoModelForSequenceClassification.from_pretrained('aspire/acge_text_embedding')
rerank_model.cuda()
rerank_model.eval()

pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

for query_idx in range(len(questions)):
    # 使用 BM25 检索得分
    doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))

    # 对得分进行排序并获取索引
    max_score_page_idxs = np.argsort(doc_scores)[:3]

    # top3进行重排序
    pairs = []
    for idx in max_score_page_idxs:
        pairs.append([questions[query_idx]["question"], pdf_content[idx]['content']])

    # 准备模型输入
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=3)
    print(inputs)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    # 获取最高分数的索引
    max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)

with open('submit_rerank.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)