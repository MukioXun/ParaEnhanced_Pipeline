import json
import os
import random
import re
import time
from typing import List, Dict, Optional
from openai import OpenAI
from prompt_gen import TaskDispatcher

# ================= 配置 =================
MODEL_NAME = "qwen-plus" # 建议使用更强的模型保证逻辑支点质量

# 移除 Sarcasm
BASE_STYLE_CONFIGS = [
    {"category": "age", "styles": ("Adult", "Child")},
    {"category": "gender", "styles": ("Male", "Female")},
]

EMOTION_PAIRS = [
    ("Happy", "Sad"),
    ("Excited", "Anxious"),
    ("Surprised", "Disgust"),
    ("Happy", "Angry")
]
EMOTION_CONFIGS = [{"category": "emotion", "styles": pair} for pair in EMOTION_PAIRS]

# ================= 工具函数 =================

def get_client():
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def safe_llm_call(client, prompt, json_mode=False):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 if not json_mode else 0,
            response_format={"type": "json_object"} if json_mode else None
        )
        content = resp.choices[0].message.content.strip()
        if json_mode:
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
            
            score_a = audit.get("score_a", 0)
            score_b = audit.get("score_b", 0)
            is_balanced = abs(score_a - score_b) <= 1
            is_good_enough = (score_a + score_b) >= 7
            
            if is_balanced and is_good_enough and not audit.get("is_too_generic", False):
                item["audit_info"] = audit
                return item
            
            # 对抗重写
            rewrite_prompt = f"""
            Rewrite this for {generator.dimension_name} ambiguity.
            Current: "{text}"
            Target: Must be plausible for both {generator.dimension_name} branches.
            Auditor Hint: {audit.get('suggestion')}
            Output ONLY the new sentence.
            """
            text = safe_llm_call(self.client, rewrite_prompt)
            if not text: break
        return None

    def verify_divergence(self, text, style_a, style_b, category):
        """逻辑差异化校验：要求AI基于不同身份给出截然不同的回应"""
        def get_ans(identity):
            p = f"""
            You are a helpful assistant. A user speaks to you.
            Detected User Profile: {identity} (Category: {category})
            User Text: "{text}"
            
            Instruction: Provide a natural response. You MUST tailor your advice/action based on the user's {category} status.
            If they are a child, consider safety/permission. If they are a specific gender, consider relevant advice.
            """
            return safe_llm_call(self.client, p)

        r1, r2 = get_ans(style_a), get_ans(style_b)
        
        judge_p = f"""
        Text: "{text}"
        Response to {style_a}: {r1}
        Response to {style_b}: {r2}
        
        Does the AI's response change significantly because of the user's {category}? 
        (e.g., restricted an action for a child but allowed for an adult, or gave different style advice for different genders).
        Return JSON: {{"diverged": true/false, "reason": "..."}}
        """
        res = safe_llm_call(self.client, judge_p, json_mode=True)
        return (res and res.get("diverged")), r1, r2

def one_sort_pipeline(config, client, refiner, dispatcher, target_count):
    success_count = 0
    while success_count < target_count:
        # Generator 内部现在会自动处理 Topic 选取
        gen = dispatcher.get_generator(config['category'], emotion_pair=config['styles'])
        
        print(f"\n[Batching] {config['category']} | {config['styles']} | Topic: {gen.current_topic}")
        
        raw_items = safe_llm_call(client, gen.build_prompt(3), json_mode=True)
        if not raw_items or not isinstance(raw_items, list): continue

        for item in raw_items:
            refined = refiner.refine_item(gen, item)
            if not refined: continue

            is_good, r1, r2 = refiner.verify_divergence(
                refined['text'], config['styles'][0], config['styles'][1], config['category']
            )
            
            if is_good:
                refined.update({
                    "category": config['category'],
                    "styles": config['styles'],
                    "responses": {config['styles'][0]: r1, config['styles'][1]: r2},
                    "topic": gen.current_topic,
                })
                success_count += 1
                yield refined
                print(f"  ✅ [{success_count}/{target_count}] Saved: {refined['text']}")
            
            if success_count >= target_count: break

# ================= 主程序 =================

def run_v6(base_target=2, emotion_target=2, output_file="dataset_v6.json"):
    client = get_client()
    refiner = DataRefiner(client)
    dispatcher = TaskDispatcher()
    all_data = []

    def process(configs, target):
        nonlocal all_data
        for config in configs:
            for item in one_sort_pipeline(config, client, refiner, dispatcher, target):
                all_data.append(item)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)

    process(BASE_STYLE_CONFIGS, base_target)
    process(EMOTION_CONFIGS, emotion_target)
    print(f"\n🎉 Done! Total: {len(all_data)}")

if __name__ == "__main__":
    run_v6(base_target=2, emotion_target=2)