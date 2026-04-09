import json
import os
import re
import sys
from typing import List, Dict, Optional
from openai import OpenAI
from prompt_gen import TaskDispatcher

# ================= 配置 =================
MODEL_NAME = "qwen-plus"

EMOTION_PAIRS = [
    ("Happy", "Sad"),
    ("Excited", "Anxious"),
    ("Surprised", "Disgust"),
    ("Happy", "Angry")
]

BASIC_PAIRS = [
        ("age", ("Adult", "Child")),
        ("gender", ("Male", "Female"))
]

# ================= 工具函数 (流式增强) =================

def get_client():
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def safe_llm_call(client, prompt, json_mode=False, stream_display=False, label="AI"):
    """
    支持流式显示的 LLM 调用工具
    :param stream_display: 是否实时打印输出内容
    :param label: 流式输出时的前缀标签
    """
    try:
        if stream_display and not json_mode:
            print(f"  \033[94m[{label}]\033[0m ", end="", flush=True)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                stream=True
            )
            full_content = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_content += content
                sys.stdout.write(content)
                sys.stdout.flush()
            print() # 换行
            return full_content.strip()
        else:
            # 非流式或 JSON 模式
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 if not json_mode else 0,
                response_format={"type": "json_object"} if json_mode else None,
                stream=False
            )
            content = resp.choices[0].message.content.strip()
            if json_mode:
                clean_content = re.sub(r"```json\s?|\s?```", "", content)
                return json.loads(clean_content)
            return content
    except Exception as e:
        print(f"\n  \033[91m[Error] LLM Call Failed: {e}\033[0m")
        return None

# ================= 核心 Pipeline 逻辑 =================

class PipelineManager:
    def __init__(self, client):
        self.client = client
        self.dispatcher = TaskDispatcher()

    def brainstorm_scenarios(self, generator):
        """流式展示灵感脑暴过程"""
        print(f"\n💡 \033[1m正在为 {generator.dimension_name} 脑暴新颖场景...\033[0m")
        prompt = generator.get_meta_prompt()
        res = safe_llm_call(self.client, prompt, stream_display=True, label="Brainstorm")
        scenarios = [line.strip() for line in res.split('\n') if len(line.strip()) > 5]
        return scenarios[:10]

    def process_and_verify(self, item, gen, styles, category):
        text = item["text"]
        style_a, style_b = styles
        
        print(f"\n\033[33m测试文本: \"{text}\"\033[0m")

        def get_ans(identity):
            system_instruction = f"""
            # Role
            你是一个专业内容生成专家。请根据用户身份({identity})提供口语化回复。
            
            # Detailed Rules
            1. [Age]: 若身份是Child，涉及敏感/成人领域(法律、安全、复杂金融)必须拒绝细节并要求咨询家长；若Adult则正常回答。
            2. [Gender]: 涉及健康，出行，衣着和饰品偏向裙装，要重点考虑男女差异；同时注意社交安全边界，女性可能需要额外的安全提醒。
            3. [Emotion]: 根据指定的({identity})语气回复（例如：Happy采取惊喜/赞赏, Angry采取宽慰/抱歉）。
            """
            # 流式输出身份回复，方便对比
            return safe_llm_call(
                self.client, 
                f"{system_instruction}\n\nUser: {text}", 
                stream_display=True, 
                label=f"Response as {identity}"
            )

        r1 = get_ans(style_a)
        r2 = get_ans(style_b)
        
        if not r1 or not r2: return None

        # 判定过程不需要流式，直接看结果
        judge_p = f"""
        用户输入: "{text}"
        身份A({style_a})回复: "{r1}"
        身份B({style_b})回复: "{r2}"
        判断维度: {category}
        返回 JSON: {{"diverged": true/false, "reason": "..."}}
        """
        res = safe_llm_call(self.client, judge_p, json_mode=True)
        
        if res and res.get("diverged"):
            print(f"  \033[92m✅ 判定通过: {res.get('reason')}\033[0m")
            item.update({
                "category": category,
                "styles": styles,
                "responses": {style_a: r1, style_b: r2},
                "logic_reasoning": res.get("reason")
            })
            return item
        else:
            print(f"  \033[90m❌ 判定失败: 差异度不足\033[0m")
            return None

    def run_dimension(self, category, styles, target_count):
        gen = self.dispatcher.get_generator(category, emotion_pair=styles)
        # 1. 第一阶段：脑暴场景（流式）
        contexts = self.brainstorm_scenarios(gen)
        
        results = []
        while len(results) < target_count:
            # 2. 第二阶段：批量生成文本（静默）
            print(f"  \033[94m[Batching] 正在生成 {category} 测试项...\033[0m", end="\r")
            batch_prompt = gen.build_prompt(count=3, brainstormed_contexts=contexts)
            raw_items = safe_llm_call(self.client, batch_prompt, json_mode=True)
            if not raw_items: continue
            
            for item in raw_items:
                # 3. 第三阶段：流式验证每一个生成项
                verified = self.process_and_verify(item, gen, styles, category)
                if verified:
                    results.append(verified)
                if len(results) >= target_count: break
        return results

# ================= 主程序 =================

def run_v7(base_target=2, emotion_target=2, output_file="dataset_v6.json"):
    client = get_client()
    manager = PipelineManager(client)
    all_data = []

    # 基础维度
    for cat, styles in BASIC_PAIRS:
        print(f"\n" + "="*50 + f"\n🚀 启动维度: {cat.upper()}\n" + "="*50)
        data = manager.run_dimension(cat, styles, base_target)
        all_data.extend(data)

    # 情感维度
    for pair in EMOTION_PAIRS:
        print(f"\n" + "="*50 + f"\n🎭 启动情感对: {pair}\n" + "="*50)
        data = manager.run_dimension("emotion", pair, emotion_target)
        all_data.extend(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 流程结束! 总计存盘: {len(all_data)} 条。")

if __name__ == "__main__":
    run_v7(base_target=2, emotion_target=1) # 情感较多，先每个对子测1条