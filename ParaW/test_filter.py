#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本中性过滤器 - 小批量测试脚本
"""

import json
import os
from openai import OpenAI


def test_qwen_api():
    """测试 Qwen API 连接"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 测试文本样本
    test_samples = [
        {
            "text": "I want to travel to Tibet alone.",
            "category": "age",
            "topic": "Travel"
        },
        {
            "text": "Can you recommend a formal interview outfit?",
            "category": "gender",
            "topic": "Fashion"
        },
        {
            "text": "My colleague got the big project.",
            "category": "emotion",
            "topic": "Work"
        },
        {
            "text": "I'm thinking about joining a fitness class.",
            "category": "gender",
            "topic": "Fitness"
        },
        {
            "text": "I want to open a bank account.",
            "category": "age",
            "topic": "Finance"
        }
    ]

    results = []

    for sample in test_samples:
        print(f"\n{'='*60}")
        print(f"测试文本: {sample['text']}")
        print(f"类别: {sample['category']}")

        prompt = build_prompt(sample)

        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一位专业的语言学分析专家，擅长识别文本中的属性敏感性。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            print(f"\n模型响应:\n{result_text}")

            # 尝试解析 JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            try:
                parsed = json.loads(result_text.strip())
                if parsed.get("is_neutral"):
                    results.append({
                        "text": sample["text"],
                        "category": sample["category"],
                        "topic": sample["topic"],
                        "styles": parsed.get("styles", []),
                        "branch_1_logic": parsed.get("branch_1_logic", ""),
                        "branch_2_logic": parsed.get("branch_2_logic", "")
                    })
            except json.JSONDecodeError:
                print("JSON 解析失败")

        except Exception as e:
            print(f"错误: {e}")

    # 保存结果
    print(f"\n{'='*60}")
    print(f"测试完成! 共筛选出 {len(results)} 条中性文本")
    print(f"\n筛选结果:")
    for r in results:
        print(f"  - [{r['category']}] {r['text'][:50]}...")

    return results


def build_prompt(text_info: dict) -> str:
    """构建分析提示词"""
    text = text_info["text"]
    category = text_info["category"]

    category_guidelines = {
        "age": """
【年龄维度分析】
判定标准：文本涉及法律责任、安全风险、认知能力或社会角色差异。
差异化逻辑：
- 成年人：独立决策、承担法律后果、专业职场背景
- 儿童：需监护人陪同、认知受限、优先保障安全""",

        "gender": """
【性别维度分析】
判定标准：文本涉及生理差异、安全感差异、特定社会礼仪或健康/着装建议。
差异化逻辑：
- 男性：男性特有的健康建议、社交礼仪
- 女性：女性特有的健康建议、安全感关怀、社交礼仪""",

        "emotion": """
【情绪维度分析】
判定标准：文本语义中性或存在多义性，在未明确说话人情感时，错误的回复会导致共情失败。
差异化逻辑：
- 正面情感 (如 Happy)：采取惊喜、赞赏、鼓励的策略
- 负面情感 (如 Angry/Sad)：采取宽慰、抱歉、安抚或解决问题的策略"""
    }

    return f"""请分析以下文本，判断它是否属于需要差异化回复的中性文本。

{category_guidelines.get(category, "")}

待分析文本：
"{text}"

请按以下 JSON 格式输出结果。如果该文本需要差异化回复，输出：
{{
    "is_neutral": true,
    "reason": "简要说明为什么需要差异化回复",
    "styles": ["属性1", "属性2"],
    "branch_1_logic": "第一种属性下的理解和回复策略",
    "branch_2_logic": "第二种属性下的理解和回复策略"
}}

如果该文本不需要差异化回复，输出：
{{
    "is_neutral": false,
    "reason": "简要说明为什么不需要差异化回复"
}}

注意：只输出 JSON 格式，不要有其他内容。"""


if __name__ == "__main__":
    # 从环境变量或直接设置 API Key
    api_key=os.getenv("DASHSCOPE_API_KEY"), 

    if not api_key:
        print("请设置 QWEN_API_KEY 环境变量，或直接在脚本中设置")
        print("示例: export QWEN_API_KEY='your_api_key'")
        # 如果想直接设置，取消下面这行的注释：
        # api_key = "sk-your-api-key-here"

    if api_key:
        test_qwen_api()
    else:
        print("未提供 API Key，退出")
