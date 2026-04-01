"""
语音-语言对齐训练数据生成脚本
用于生成依赖声学信息（副语言）才能正确理解的对话查询数据
"""

import json
import os
import random
import time
from typing import Optional
from openai import OpenAI


# ============== 配置 ==============

# 风格类型定义
EMOTION_CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
STYLE_CATEGORIES = {
    "Emotion": [
        ("语气兴奋，语速较快，声音明亮，带有愉悦感", "语气低落，语速缓慢，声音略显疲惫，带有沉重感"),
        ("语气愤怒，声音尖锐，语速快而急促", "语气平静，声音柔和，语速均匀稳定"),
        ("语气紧张，声音颤抖，语速不稳", "语气放松，声音沉稳，语速从容"),
        ("语气开心，充满活力，声调上扬", "语气悲伤，声音低沉，带着哭腔"),
    ],
    "Emotion_Polarity": [
        ("happy", "sad"),
        ("surprised", "fearful"),
        ("happy", "disgusted"), # 这种对比常产生讽刺
    ],
    "Age": [
        ("语气年轻活泼，充满朝气", "语气成熟稳重，声音深沉有力"),
        ("语气天真烂漫，充满好奇", "语气老成持重，声音沧桑"),
    ],
    "Gender": [
        ("男性特征：声音低沉，语速沉稳", "女性特征：声音柔美，语调更富变化", "男性", "女性"),
    ],
    "Sarcasm": [
        ("真诚赞赏：语气诚恳，充满敬意，表达由衷的认可", "极度讽刺：反语手法，语调夸张且带嘲弄，‘庄词谐用’（用极其隆重的词汇描述微不足道的错误）"),
        ("平实陈述：语气客观，不带个人偏见", "阴阳怪气：语速缓慢，重音诡异，通过夸张的赞美来表达强烈不满"),
    ],
}

# 主题类别
TOPICS = [
    "Hobby",      # 兴趣
    "Work",       # 工作
    "Study",      # 学习
    "Relationship",  # 人际关系
    "Travel",     # 旅行
    "Health",     # 健康
    "Fashion",    # 时尚
    "Finance",    # 财务
]

# 主题中文映射
TOPIC_CN = {
    "Hobby": "兴趣爱好",
    "Work": "工作职场",
    "Study": "学习教育",
    "Relationship": "人际关系",
    "Travel": "旅行出行",
    "Health": "健康生活",
    "Fashion": "时尚穿搭",
    "Finance": "财务理财",
}


# ============== API 客户端 ==============

def get_client() -> OpenAI:
    """获取 OpenAI 兼容客户端"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# ============== 数据生成 ==============
def build_prompt(topic: str, style_category: str, count: int = 3) -> str:
    prompt = f"""你是一个顶级的多模态语料专家。你需要生成一种特殊的训练数据：**其文本是极度中立的“空壳”，而灵魂（意图/身份/修辞）完全取决于副语言。**

### 1. 核心挑战
- **属性隐身术（Age/Gender）**：严禁在文本中出现任何指代年龄（如“老了”、“小孩”）或性别（如“姐”、“哥”）的词汇。
- **高级修辞陷阱（Sarcasm/Irony）**：利用【反语】、【夸张】或【庄词谐用】。
  - *庄词谐用示例*：文本是“您的这一举动真可谓是惊天动地，旷古烁今”，如果语音是讽刺的，那它就是在嘲笑对方犯了低级错误；如果是真诚的，则是在赞美。

### 2. 任务要求
- **当前主题**：{TOPIC_CN.get(topic, topic)}
- **风格维度**：{style_category}
- **自由探索**：请构思复杂的社会互动场景（如：职场汇报、商场退货、老友重逢、医患沟通）。

### 3. 生成准则
1. **文本模糊性**：文本必须在两种风格下都能自洽，但指向完全相反的后果。
2. **决策差异化**：
   - 如果是【年龄/性别】，AI的回复必须根据“听出来的身份”给出不同的专业建议或互动逻辑。
   - 如果是【讽刺/反语】，AI必须识破对方的“正话反说”，给出对应的（如道歉、解释或回击）回复。

### 4. 输出格式 (JSON)
```json
[
  {{
    "scenario": "<隐藏背景：例如用户在评价一个糟糕的方案>",
    "text": "<文本内容：如‘你的这个构思确实超出了我的想象空间。’>",
    "style_1": "<真诚风格描述>",
    "style_2": "<讽刺/庄词谐用风格描述>",
    "expected_difference": "<风格1下：AI应顺着赞美深度探讨；风格2下：AI应意识到用户在表达不满，需立刻询问改进点。>"
  }}
]
"""
    return prompt


def generate_batch(
    client: OpenAI,
    topic: str,
    style_category: str,
    count: int = 3,
    model: str = "qwen-plus"
) -> list[dict]:
    """生成一批数据"""
    prompt = build_prompt(topic, style_category, count)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )

        content = completion.choices[0].message.content

        # 提取JSON部分
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
        return data

    except Exception as e:
        print(f"生成失败: {e}")
        return []


# ============== 数据验证 ==============

def validate_item(item: dict) -> tuple[bool, str]:
    """验证单条数据是否符合要求"""

    # 检查必要字段
    required_fields = ["text", "style_1", "style_2", "expected_difference", "topic"]
    for field in required_fields:
        if field not in item or not item[field]:
            return False, f"缺少字段: {field}"

    # 1. 副语言相关性检查：expected_difference 应该描述明显不同的回应
    diff = item["expected_difference"]
    if len(diff) < 10:
        return False, "expected_difference 过短，未能说明回应差异"

    # 2. 中立性检查：文本不应包含明显情绪词
    text = item["text"]
    emotion_keywords = [
        "太棒", "太好了", "气死", "开心", "难过", "愤怒", "兴奋",
        "惊喜", "失望", "绝望", "崩溃", "幸福", "痛苦", "！",
        "好烦", "讨厌", "喜欢", "爱死", "恨", "呜呜", "哈哈",
        "！", "？！", "！！"
    ]
    for keyword in emotion_keywords:
        if keyword in text:
            return False, f"文本包含情绪词: {keyword}"

    # 3. 合理性检查：文本长度适中
    if len(text) < 5:
        return False, "文本过短"
    if len(text) > 100:
        return False, "文本过长"

    # 4. 风格对比检查：style_1 和 style_2 应该有差异
    if item["style_1"] == item["style_2"]:
        return False, "两种风格相同"

    return True, "通过"


def filter_and_validate(data: list[dict]) -> list[dict]:
    """过滤和验证数据"""
    valid_data = []
    for item in data:
        is_valid, reason = validate_item(item)
        if is_valid:
            valid_data.append(item)
        else:
            print(f"  过滤: {reason} - {item.get('text', 'N/A')[:30]}")
    return valid_data


# ============== 主流程 ==============

def generate_dataset(
    total_count: int = 100,
    output_file: str = "sft_paralinguistic_data.json",
    model: str = "qwen-plus",
    batch_size: int = 5
) -> list[dict]:
    """
    生成完整数据集

    Args:
        total_count: 目标生成数量
        output_file: 输出文件路径
        model: 使用的模型
        batch_size: 每批生成数量
    """
    client = get_client()
    all_data = []

    # 计算每个主题的目标数量
    per_topic = total_count // len(TOPICS)
    style_categories = list(STYLE_CATEGORIES.keys())

    print(f"目标生成 {total_count} 条数据")
    print(f"每个主题约 {per_topic} 条")
    print("=" * 50)

    for topic in TOPICS:
        print(f"\n生成主题: {TOPIC_CN[topic]}")
        topic_data = []
        attempts = 0
        max_attempts = per_topic * 3  # 最多尝试3倍次数

        while len(topic_data) < per_topic and attempts < max_attempts:
            attempts += 1

            # 随机选择风格类别
            style_category = random.choice(style_categories)

            print(f"  尝试 {attempts}: 风格={style_category}", end=" ")

            # 生成一批数据
            batch = generate_batch(
                client=client,
                topic=topic,
                style_category=style_category,
                count=batch_size,
                model=model
            )

            # 验证和过滤
            valid_batch = filter_and_validate(batch)
            topic_data.extend(valid_batch)

            print(f"-> 获得有效数据 {len(valid_batch)} 条 (累计: {len(topic_data)})")

            # 避免API限流
            time.sleep(0.5)

        all_data.extend(topic_data[:per_topic])
        print(f"主题 {TOPIC_CN[topic]} 完成: {len(topic_data[:per_topic])} 条")

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"生成完成！共 {len(all_data)} 条数据")
    print(f"已保存到: {output_file}")

    # 统计主题分布
    topic_dist = {}
    for item in all_data:
        t = item.get("topic", "Unknown")
        topic_dist[t] = topic_dist.get(t, 0) + 1

    print("\n主题分布:")
    for t, c in topic_dist.items():
        print(f"  {TOPIC_CN.get(t, t)}: {c} 条")

    return all_data


def generate_single_batch(
    topic: str = None,
    style_category: str = None,
    count: int = 5,
    model: str = "qwen-plus"
) -> list[dict]:
    """
    生成单批数据（用于测试或增量生成）

    Args:
        topic: 指定主题，为空则随机
        style_category: 指定风格类别，为空则随机
        count: 生成数量
        model: 使用的模型
    """
    client = get_client()

    if topic is None:
        topic = random.choice(TOPICS)
    if style_category is None:
        style_category = random.choice(list(STYLE_CATEGORIES.keys()))

    print(f"生成数据: 主题={TOPIC_CN[topic]}, 风格={style_category}, 数量={count}")

    data = generate_batch(
        client=client,
        topic=topic,
        style_category=style_category,
        count=count,
        model=model
    )

    valid_data = filter_and_validate(data)

    print(f"有效数据: {len(valid_data)} 条")
    for item in valid_data:
        print(f"  - {item['text']}")

    return valid_data


# ============== 入口 ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成语音-语言对齐训练数据")
    parser.add_argument("-n", "--count", type=int, default=50, help="生成数据总数")
    parser.add_argument("-o", "--output", type=str, default="sft_paralinguistic_data.json", help="输出文件")
    parser.add_argument("-m", "--model", type=str, default="qwen-plus", help="使用的模型")
    parser.add_argument("--test", action="store_true", help="测试模式，生成少量数据")

    args = parser.parse_args()

    if args.test:
        # 测试模式
        print("=== 测试模式 ===")
        data = generate_single_batch(count=3, model=args.model)
        print("\n生成的数据:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        # 正式生成
        generate_dataset(
            total_count=args.count,
            output_file=args.output,
            model=args.model
        )
