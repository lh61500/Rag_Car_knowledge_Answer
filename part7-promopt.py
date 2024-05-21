import time
import requests
import jwt
#from zhipuAI import ZhipuAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# zhipuAI官网 http调用例程
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )
#这里是将用户询问content传到gpt进行回答
def ask_glm(content):
    #client=ZhipuAI(api_key="ad7d1576d2af756b50cd5db221b988b5.HOdZN9RuKF9N7iFf")
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token("ad7d1576d2af756b50cd5db221b988b5.HOdZN9RuKF9N7iFf", 1000)
    }

    data = {
        "model": "glm-4",
        "messages": [{"role": "user", "content": content}]
    }
    #这里
    '''这是zhipuAI的标准模板
    curl - -location
    'https://open.bigmodel.cn/api/paas/v4/chat/completions' \
    - -header
    'Authorization: Bearer <你的apikey>' \
    - -header
    'Content-Type: application/json' \
    - -data
    '{
    "model": "glm-4",
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ]

    }'
    '''
    #这里request。post()是有确定格式的 post(url,headers,json)三个参数
    response = requests.post(url, headers=headers, json=data)
    return response.json()
#a=ask_glm('你好')
#print(a)
#这是语义化模型
tokenizer = AutoTokenizer.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
rerank_model = AutoModelForSequenceClassification.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
rerank_model.cuda()
rerank_model.eval()

