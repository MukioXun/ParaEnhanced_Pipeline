import json
import os
import random
import re
import time
from typing import List, Dict, Optional
from openai import OpenAI

# ================= 1. 配置与标签体系 =================

TOPICS = ["Hobby", "Work", "Study", "Relationship", "Travel", "Health", "Fashion", "Finance"]
TOPIC_CN = {
    "Hobby": "兴趣爱好", "Work": "工作职场", "Study": "学习教育",
    "Relationship": "人际关系", "Travel": "旅行出行", "Health": "健康生活",
    "Fashion": "时尚穿搭", "Finance": "财务理财"
}

# 情感标签库
EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# 风格对配置：包含标签、描述、修辞手法倾向及是否涉及身份属性
STYLE_CONFIGS = [
    {
        "category": "Rhetoric_Sarcasm",
        "pair": ("happy", "disgusted"),
        "desc": ("真诚赞赏：语气热忱，表达由衷认可", "庄词谐用：用极其华丽隆重的词汇描述微不足道的错误，语气轻蔑讽刺"),
        "is_identity": False
    },
    {
        "category": "Irony_Contrast",
        "pair": ("neutral", "angry"),
        "desc": ("客观平实陈述", "反语讽刺：通过夸张的委婉语表达强烈不满，语速缓慢，重音诡异"),
        "is_identity": False
    },
    {
        "category": "Gender_Logic",
        "pair": ("neutral", "neutral"),
        "desc": ("成熟男性：声音低沉，语速沉稳，透着果断", "年轻女性：声音柔美，语调更富变化，富有同理心"),
        "is_identity": True
    },
    {
        "category": "Age_Logic",
        "pair": ("fearful", "neutral"),
        "desc": ("稚嫩儿童：语调高昂，带着局促与不安", "资深专家：语气从容，不紧不慢，透着‘过来人’的淡定"),
        "is_identity": True
    },
    {
        "category": "Emotion_Polarity",
        "pair": ("surprised", "sad"),
        "desc": ("惊喜：音调上扬，充满期待", "极度失落：声音低沉，带有疲惫感和破碎感"),
        "is_identity": False
    }
]

MODEL_NAME = "qwen-plus" # 或使用 gpt-4o 等

# ================= 2. API 核心逻辑 =================

def get_client():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key: raise ValueError("请设置 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def llm_call(client, prompt, temperature=0.7, json_mode=False):
    try:
        # 如果开启了 json_mode，确保 prompt 中包含 "json" 单词，否则 API 会报错 400
        if json_mode and "json" not in prompt.lower():
            prompt += "\n\n请务必以 JSON 格式返回结果。"
            
        response_format = {"type": "json_object"} if json_mode else None
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format=response_format
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # 打印更详细的错误，方便定位
        print(f"API Error: {e}")
        return None

# ================= 3. 数据生成模块 =================

def build_generate_prompt(topic: str, config: Dict, count: int):
    label1, label2 = config["pair"]
    desc1, desc2 = config["desc"]
    
    identity_logic = ""
    if config["is_identity"]:
        identity_logic = """
- **属性隐身原则**：严禁在文本中出现性别（男、女、姐、哥）或年龄（老、少）等身份指代词。
- **决策偏移要求**：分支1和分支2的 expected_response 必须体现出针对不同身份的【实质性建议差异】（如：推荐完全不同的理财产品或服装款式）。"""

    return f"""你是一个高级语料专家。请在主题【{TOPIC_CN[topic]}】下生成 {count} 条高度依赖副语言的对话数据。

### 核心任务
创建一个【绝对中立】的文本作为“谜面”，通过两种不同的【副语言风格】（谜底）引导出截然不同的AI回复逻辑。

### 风格约束
- 分支1 (标签:{label1})：{desc1}
- 分支2 (标签:{label2})：{desc2}
- **修辞手法**：若涉及讽刺，请使用“庄词谐用”（用极正式/隆重的词描述糟糕的事）或“反语”。
{identity_logic}

### 输出格式 (JSON Array)
[
  {{
    "scenario": "详细描述隐藏背景",
    "text": "中立文本（严禁泄露情绪、身份、态度）",
    "branch_1": {{ "label": "{label1}", "style": "{desc1}", "expected_logic": "识别到此风格后，AI的回复策略" }},
    "branch_2": {{ "label": "{label2}", "style": "{desc2}", "expected_logic": "识别到此风格后，AI的完全反转/不同的回复策略" }}
  }}
]
"""

# ================= 4. 验证与重写模块 =================

def validate_neutrality(text: str) -> bool:
    """硬核过滤显式提示词"""
    blacklist = ["真棒", "太烂", "生气", "开心", "难过", "我一个男", "我一个女", "我老了", "小孩", "垃圾", "极好"]
    return not any(word in text for word in blacklist)

def judge_ambiguity(client, text: str):
    """通过LLM判断文本是否具有足够的歧义性"""
    prompt = f"分析此文本：'{text}'。在没有任何语音语调的情况下，你能确定说话者的真实情绪或身份吗？请回答：{{'is_ambiguous': true/false, 'leakage': '描述泄露点'}}"
    res = llm_call(client, prompt, temperature=0, json_mode=True)
    try: return json.loads(res)
    except: return {"is_ambiguous": False}

def adversarial_rewrite(client, text: str):
    """对抗重写：移除文本中的情绪/身份线索，使其变为空壳"""
    prompt = f"请改写这句话，移除所有表达情绪、年龄、性别的显式词汇，使其变得极度中立且依赖语音语调才能理解真实意图：\n原句：{text}\n仅输出改写后的句子。"
    return llm_call(client, prompt, temperature=0.8)

# ================= 5. 响应差异化验证 =================

def verify_response_divergence(client, text, b1, b2):
    """模拟两个分支的真实回复，并计算逻辑差异"""
    def get_resp(style):
        p = f"用户说：'{text}'\n（语音背景：{style}）\n请给出你的简短回复："
        return llm_call(client, p, temperature=0.7)

    r1 = get_resp(b1["style"])
    r2 = get_resp(b2["style"])
    
    # 让LLM评估两个回复是否真的发生了“逻辑转向”
    judge_p = f"文本：'{text}'\n回复A: {r1}\n回复B: {r2}\n这两个回复在建议内容、意图理解或情感基调上是否具有【本质差异】？输出：{{'diverged': true/false, 'reason': '...'}}"
    res = llm_call(client, judge_p, temperature=0, json_mode=True)
    try:
        data = json.loads(res)
        return data.get("diverged", False), r1, r2
    except:
        return False, r1, r2

# ================= 6. 主 Pipeline =================

def run_v3_pipeline(total_count=10, output_file="paralinguistic_v3.json"):
    client = get_client()
    final_dataset = []
    
    print(f"🚀 开始 Pipeline v3，目标生成 {total_count} 条高质量对齐数据...")

    while len(final_dataset) < total_count:
        topic = random.choice(TOPICS)
        config = random.choice(STYLE_CONFIGS)
        
        # 1. 生成初始批次
        raw_data = llm_call(client, build_generate_prompt(topic, config, 3))
        if not raw_data: continue
        
        # 尝试从Markdown中提取JSON
        items = []
        try:
            match = re.search(r"\[.*\]", raw_data, re.S)
            if match: items = json.loads(match.group())
        except: continue

        for item in items:
            text = item["text"]
            
            # 2. 中立性与歧义性校验
            if not validate_neutrality(text):
                text = adversarial_rewrite(client, text)
            
            ambiguity = judge_ambiguity(client, text)
            if not ambiguity.get("is_ambiguous"):
                continue # 泄露严重的直接丢弃
            
            item["text"] = text # 更新重写后的文本
            
            # 3. 验证逻辑响应是否真的发生了偏移
            diverged, r1, r2 = verify_response_divergence(client, text, item["branch_1"], item["branch_2"])
            
            if diverged:
                item["topic"] = topic
                item["verification"] = {"resp_1": r1, "resp_2": r2}
                final_dataset.append(item)
                print(f"✅ [{len(final_dataset)}] 成功入库: {text[:20]}... (类型: {config['category']})")
            else:
                print(f"❌ 过滤: 响应差异过小 - {text[:20]}")

            if len(final_dataset) >= total_count: break
        
        time.sleep(1) # 频率限制

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 任务完成！数据已保存至 {output_file}")

if __name__ == "__main__":
    # 示例运行
    run_v3_pipeline(total_count=5)