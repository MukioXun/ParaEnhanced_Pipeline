你需要构建用于语音-语言对齐训练（SFT）的数据。目标是生成依赖声学信息（副语言）才能正确理解的对话查询。

🧠 核心设计原则（必须严格遵守）
对比风格原则（Contrastive Style）
每条查询必须配对两种语义相同但说话风格相反的表达：
风格差异必须足够显著（如：开心 vs 失落、真诚 vs 讽刺）
两种风格必须导致合理但明显不同的回复
文本中立性原则（Text Neutrality）
文本内容必须不包含任何显式情绪/态度/身份线索
模型不能仅依赖文本推断说话状态
状态必须只能通过语音风格识别
副语言可影响性原则（Paralinguistic Sensitivity）
查询必须属于“语气会影响回应”的场景
禁止生成对语气不敏感的问题（如事实问答）
🏗️ 数据生成格式（严格按 JSON 输出）

每条数据必须符合以下结构：

{
  "text": "<中立文本内容>",
  "style_1": "<风格1描述>",
  "style_2": "<风格2描述>",
  "expected_difference": "<两种风格应导致的回复差异说明>",
  "topic": "<主题类别>"
}
🎨 风格类型定义（按任务选择）

根据任务类型，从以下类别中选择风格对：

Emotion（情感）：开心 / 悲伤 / 愤怒 / 紧张 / 平静
Sarcasm（讽刺）：真诚 / 讽刺
Age（年龄）：年轻 / 年长
Gender（性别）：男性 / 女性
🌍 主题覆盖要求

生成内容必须均匀覆盖以下主题：

兴趣（Hobby）
工作（Work）
学习（Study）
人际关系（Relationship）
旅行（Travel）
健康（Health）
时尚（Fashion）
财务（Finance）
⚠️ 自动过滤标准（生成时自检）

生成的每条数据必须满足以下三项，否则直接丢弃：

副语言相关性检查
两种风格是否会导致明显不同的回应？
❌ 错误示例：事实性问题（如“美国总统是谁”）
合理性检查
文本与风格组合是否现实合理？
❌ 错误示例：不符合身份/常识的组合
中立性检查
文本是否泄露情绪/态度/身份？
❌ 错误示例：
“太棒了！”（明显情绪）
“我气死了”（直接情绪表达）
🔁 生成任务

请生成 N 条数据（每条包含一对对比风格），要求：

风格对具有强对比性
每条数据必须通过上述三项检查
保证主题分布均衡
避免重复表达
📌 示例（高质量参考）
{
  "text": "我刚刚遇到了一件挺特别的事情。",
  "style_1": "语气兴奋，语速较快，带明显上扬语调",
  "style_2": "语气低落，语速缓慢，声音略显疲惫",
  "expected_difference": "前者应引导积极倾听和兴趣表达，后者应表现出关心与安慰",
  "topic": "Relationship"
}

使用Qwen的api实现，下面是api的使用例子
from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "你是谁"}]
completion = client.chat.completions.create(
    model="qwen3.5-flash",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
is_answering = False  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20)
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)
