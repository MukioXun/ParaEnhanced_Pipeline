import json
import os
import random
import re
import time
from typing import List, Dict, Optional
from openai import OpenAI
from prompt_gen import TaskDispatcher

# ================= 配置 =================
MODEL_NAME = "qwen3.5-flash"
# 基础维度
BASE_STYLE_CONFIGS = [
    {"category": "sarcasm", "styles": ("Sincere", "Sarcastic")},
    {"category": "age", "styles": ("Adult", "Child")},
    {"category": "gender", "styles": ("Male", "Female")},
]
# 细分情绪对
EMOTION_PAIRS = [
    ("Happy", "Sad"),
    ("Excited", "Anxious"),
    ("Surprised", "Disgust"),
    ("Sad", "Fear"),
    ("Happy", "Angry")
]
EMOTION_CONFIGS = [{"category": "emotion", "styles": pair} for pair in EMOTION_PAIRS]
TOPICS = [
    "Work & Studies", "Money & Transactions", "Health",
    "Personal Life", "Fashion & Style", "Entertainment"
]

# ================= 工具函数 =================

def get_client():
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def safe_llm_call(client, prompt, json_mode=False):
    """鲁棒的LLM调用，包含自动JSON修复"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 if not json_mode else 0, # JSON模式使用0以保证稳定
            response_format={"type": "json_object"} if json_mode else None
        )
        content = resp.choices[0].message.content.strip()
        
        if json_mode:
            # 尝试清洗 Markdown 代码块
            clean_content = re.sub(r"```json\s?|\s?```", "", content)
            return json.loads(clean_content)
        return content
    except Exception as e:
        print(f"  [Error] LLM Call Failed: {e}")
        return None

# ================= 核心 Pipeline 逻辑 =================
class DataRefiner:
    def __init__(self, client):
        self.client = client

    def refine_item(self, generator, item: Dict) -> Optional[Dict]:
        text = item["text"]
        for i in range(2): 
            audit = safe_llm_call(self.client, generator.build_critic_prompt(text), json_mode=True)
            if not audit: break
            # 核心优化逻辑：
            # 1. 总分大于等于 7 即可接受（不再追求完美的10分）
            # 2. 两个分支的分数差异不能超过 1 分（确保平衡）
            # 3. 拒绝过于无聊的句子（Generic Check）
            score_a = audit.get("score_a", 0)
            score_b = audit.get("score_b", 0)
            is_balanced = abs(score_a - score_b) <= 1
            is_good_enough = (score_a + score_b) >= 7
            
            if is_balanced and is_good_enough and not audit.get("is_too_generic", False):
                item["text"] = text
                item["audit_info"] = audit
                return item
            # 如果不通过，利用审计员的 suggestion 进行重写
            print(f"    - [Refining] Balance: {score_a} vs {score_b} | Generic: {audit.get('is_too_generic')}")
            rewrite_prompt = f"""
            Rewrite this sentence to be more balanced and evocative.
            Current Text: "{text}"
            Auditor Suggestion: {audit.get('suggestion')}
            Ensure it fits BOTH interpretations of {generator.dimension_name} equally well.
            Just output the sentence.
            """
            text = safe_llm_call(self.client, rewrite_prompt)
            if not text: break
            
        return None


    def verify_divergence(self, text, style_a, style_b):
        """逻辑差异化校验：针对情绪对进行强化"""
        def get_ans(s):
            # 强化指令：要求 AI 必须根据语音语调调整其回复策略
            p = f"""
            User said: "{text}"
            The user's voice sounds: {s}
            Instruction: Provide a natural AI response. Your response MUST be tailored to the user's detected emotion. 
            If they sound {s}, adjust your empathy level, helpfulness, and tone accordingly.
            """
            return safe_llm_call(self.client, p)

        r1, r2 = get_ans(style_a), get_ans(style_b)
        
        # 评审逻辑：重点看“意图理解”是否发生了偏移
        judge_p = f"""
        Text: "{text}"
        Response A (as if user is {style_a}): {r1}
        Response B (as if user is {style_b}): {r2}
        
        TASK: Do these two responses reflect a fundamental difference in how the AI perceived the user's situation?
        Example of divergence: For 'My boss wants to see me', AI A offers congratulations (Happy), while AI B offers support/advice (Anxious).
        
        Return JSON: {{"diverged": true/false, "reason": "..."}}
        """
        res = safe_llm_call(self.client, judge_p, json_mode=True)
        return (res and res.get("diverged")), r1, r2

def one_sort_pipeline(config, client, refiner, dispatcher, target_count):
    """
    单维度生成-审计-验证流程 (生成器模式)
    """
    success_count = 0
    while success_count < target_count:
        topic = random.choice(TOPICS)
        gen = dispatcher.get_generator(config['category'], topic, emotion_pair=config['styles'])
        
        print(f"\n[Batching] Category: {config['category']} | Styles: {config['styles']} | Topic: {topic}")
        
        # 1. 批量生成初始数据
        raw_items = safe_llm_call(client, gen.build_prompt(3), json_mode=True)
        if not raw_items or not isinstance(raw_items, list):
            time.sleep(1) # 容错
            continue

        for item in raw_items:
            # 2. 审计与对抗重写
            refined = refiner.refine_item(gen, item)
            if not refined:
                continue

            # 3. 逻辑差异化验证
            is_good, r1, r2 = refiner.verify_divergence(
                refined['text'], 
                config['styles'][0], 
                config['styles'][1]
            )
            
            if is_good:
                refined.update({
                    "category": config['category'],
                    "styles": config['styles'],
                    "responses": {"style_a": r1, "style_b": r2},
                    "topic": topic,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                success_count += 1
                yield refined  # <--- 关键：流式输出
                print(f"  ✅ [Progress {success_count}/{target_count}] Success: {refined['text'][:30]}...")
            else:
                print(f"  ❌ Filtered: Response divergence too low for '{refined['text'][:20]}'")

            if success_count >= target_count:
                break
# ================= 主运行程序 =================

def run_v5(base_target=5, emotion_target=2, output_file="dataset_v5_stream.json"):
    client = get_client()
    refiner = DataRefiner(client)
    dispatcher = TaskDispatcher()
    
    all_data = []
    
    # 如果文件已存在，先读取旧数据（可选，支持断点续传）
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_data = json.load(f)
                print(f"📂 Loaded {len(all_data)} existing items from {output_file}")
        except:
            pass
    def process_and_save(configs, target):
        nonlocal all_data
        for config in configs:
            for new_item in one_sort_pipeline(config, client, refiner, dispatcher, target):
                all_data.append(new_item)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Total Items Saved: {len(all_data)}")
    print(f"🚀 Pipeline v5 Started. Target per Base: {base_target}, per Emotion Pair: {emotion_target}")
    # 1. 执行基础维度
    process_and_save(BASE_STYLE_CONFIGS, base_target)
    # 2. 执行情绪对维度
    process_and_save(EMOTION_CONFIGS, emotion_target)
    print(f"\n🎉 All Tasks Completed! Final count: {len(all_data)}")
    
if __name__ == "__main__":
    # 执行：基础类别每类5条，5对情绪每类2条，总计 15 + 10 = 25条
    run_v5(base_target=2, emotion_target=2)