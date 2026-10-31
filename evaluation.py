import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
import openai

# ================= 加载 .env 保险箱 =================
load_dotenv()

# ---------------- 魔法补丁：强行修复智谱 temperature 精度报错 ----------------
_original_create = openai.resources.chat.completions.Completions.create

def _patched_create(*args, **kwargs):
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        # 强行将温度四舍五入到两位小数
        kwargs["temperature"] = round(float(kwargs["temperature"]), 2)
    return _original_create(*args, **kwargs)

openai.resources.chat.completions.Completions.create = _patched_create
# ------------------------------------------------------------------------------

# 1. 配置智谱大模型 (RAGAs 内部需要用大模型来打分)
# 从保险箱中读取 Key
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

judge_llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.01
)

# 配置本地嵌入模型，路径指向你之前下载的 BGE 模型
# 注意：这里的路径一定要是你电脑里真实存在的绝对路径 (保持你截图里的原样即可)
local_embeddings = HuggingFaceEmbeddings(model_name="/Users/liangjiajun/Desktop/capstone_project/bge-small-zh-v1.5")

# 2. 构造评估数据集（模拟 3 个测试用例）
data_samples = {
    'question': [
        '公司的年假有几天？',
        '应该怎么申请年假？',
        '产品A的保修期是多久？'
    ],
    'contexts': [
        ['公司规定：员工年假为每年5天。'],
        ['公司规定：员工年假为每年5天，需提前一天在企业微信提交申请。'],
        ['公司产品A的保修期为1年。']
    ],
    'answer': [
        '根据公司规定，员工每年享有5天年假。',
        '需要提前一天在企业微信上提交申请。',
        '产品A的保修期是3年。'
    ],
    'ground_truth': [
        '5天',
        '提前一天在企业微信提交申请',
        '1年'
    ]
}

dataset = Dataset.from_dict(data_samples)

print("正在调用 RAGAs 评估框架，请稍候...")

# 3. 执行评估
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ],
    llm=judge_llm,
    embeddings=local_embeddings
)

# 4. 打印结果
print("\n" + "="*50)
print("RAGAs 自动化评估报告")
print("="*50)
print(result)

# 将结果保存为 CSV，方便放在 GitHub 上
df = result.to_pandas()
df.to_csv("ragas_evaluation_result.csv", index=False)
print("\n[OK] 评估结果已保存至 ragas_evaluation_result.csv")