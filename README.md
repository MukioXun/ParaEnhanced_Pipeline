# 语音-语言对齐训练数据生成流水线

## 项目概述

本项目旨在构建用于**语音-语言对齐训练（SFT）**的高质量数据集。核心目标是生成一类特殊的对话数据：**文本本身是中立的"空壳"，而真实意图、情感状态或身份属性只能通过语音的副语言特征（语调、语速、音色等）来识别**。

这类数据的训练价值在于：让多模态模型学会"听懂言外之意"，而非仅依赖文本语义进行回复。

---

## 项目结构

```
Msg_Gen/
├── README.md                    # 项目文档
├── ParaW/                       # 📝 文本生成模块
│   ├── prompt_gen.py           # Prompt模板 + 任务调度器
│   ├── run_pipeline.py         # 主流水线脚本
│   └── results/
│       └── dataset_v6.json     # 生成的文本数据集
│
├── ParaG/                       # 🎵 音频生成模块
│   ├── audio_gen.py            # TTS音频合成器 (CosyVoice3)
│   ├── voice_ref/              # 参考音频库
│   │   ├── adult_default.wav   # 成年人音色
│   │   ├── child_default.wav   # 儿童音色
│   │   ├── gender_male.wav     # 男性音色
│   │   ├── gender_female.wav   # 女性音色
│   │   ├── emo_happy.wav       # 开心情绪
│   │   ├── emo_sad.wav         # 悲伤情绪
│   │   ├── emo_angry.wav       # 愤怒情绪
│   │   ├── emo_surprised.wav   # 惊讶情绪
│   │   ├── emo_digust.wav      # 厌恶情绪
│   │   └── emo_fearful.wav     # 恐惧情绪
│   ├── voice_ref_transcriptions.json  # 参考音频转写结果
│   └── CosyVoice/              # CosyVoice 模型库
│
└── doc_lock/                    # 历史版本存档
```

---

## 模块状态

### ParaW - 文本生成模块 ✅

| 组件 | 状态 | 说明 |
|------|------|------|
| 任务调度器 | ✅ 完成 | `TaskDispatcher` 支持 Age/Gender/Emotion 三类维度 |
| 维度生成器 | ✅ 完成 | `AgeGenerator`, `GenderGenerator`, `EmotionGenerator` |
| 话题池 | ✅ 完成 | 每个维度有专属高契合度话题池 |
| 三级验证机制 | ✅ 完成 | 规则过滤 → LLM歧义判断 → 响应差异验证 |
| 对抗重写 | ✅ 完成 | 不通过时自动重写文本 |

### ParaG - 音频生成模块 ✅

| 组件 | 状态 | 说明 |
|------|------|------|
| TTS引擎 | ✅ 完成 | CosyVoice3 集成 |
| 音色字典 | ✅ 完成 | 11种风格标签 → 参考音频映射 |
| Zero-shot合成 | ✅ 完成 | 支持参考音频克隆音色 |
| 批处理流程 | ✅ 完成 | `process_dataset()` 支持批量生成 |
| 缺失音频跳过 | ✅ 完成 | 自动跳过不存在的参考音频 |

### 待补充参考音频

| 风格 | 文件名 | 状态 |
|------|--------|------|
| Excited | emo_excited.wav | ❌ 缺失 |
| Anxious | emo_anxious.wav | ❌ 缺失 |

---

## 快速开始

### 环境配置

```bash
# 1. 安装依赖
pip install openai torchaudio librosa x_transformers

# 2. 设置API密钥（文本生成需要）
export DASHSCOPE_API_KEY="your-api-key"

# 3. 确保模型已下载
# - CosyVoice3: /home/u2023112559/qix/Models/Models/CosyVoice3
# - Whisper: /home/u2023112559/qix/Models/Models/whisper-medium
```

### 文本数据生成

```bash
cd ParaW

# 运行流水线生成数据
python run_pipeline.py

# 输出: results/dataset_v6.json
```

**参数配置** (修改 `run_pipeline.py`):
```python
run_v6(
    base_target=2,      # Age + Gender 类别各生成数量
    emotion_target=2,   # 每个情感对生成数量
    output_file="dataset_v6.json"
)
```

### 音频合成

```bash
cd ParaG

# 查看可用风格
python audio_gen.py --list-styles

# 验证参考音频
python audio_gen.py --validate

# 合成音频
python audio_gen.py --input ../ParaW/results/dataset_v6.json --output audio_output
```

---

## 风格标签体系

### 当前支持的维度

| 类别 | 风格对 | 说明 |
|------|--------|------|
| **Age** | Adult ↔ Child | 年龄导致的权限/责任差异 |
| **Gender** | Male ↔ Female | 性别相关的建议差异 |
| **Emotion** | Happy ↔ Sad | 情感极性对比 |
| **Emotion** | Excited ↔ Anxious | 情感极性对比 |
| **Emotion** | Surprised ↔ Disgust | 情感极性对比 |
| **Emotion** | Happy ↔ Angry | 情感极性对比 |

### 音色字典映射

```python
VOICE_DICTIONARY = {
    # Age
    "Adult": "voice_ref/adult_default.wav",
    "Child": "voice_ref/child_default.wav",
    
    # Gender
    "Male": "voice_ref/gender_male.wav",
    "Female": "voice_ref/gender_female.wav",
    
    # Emotion
    "Happy": "voice_ref/emo_happy.wav",
    "Sad": "voice_ref/emo_sad.wav",
    "Angry": "voice_ref/emo_angry.wav",
    "Surprised": "voice_ref/emo_surprised.wav",
    "Disgust": "voice_ref/emo_digust.wav",
    "Fearful": "voice_ref/emo_fearful.wav",
    # "Excited": "voice_ref/emo_excited.wav",  # 待补充
    # "Anxious": "voice_ref/emo_anxious.wav",  # 待补充
}
```

---

## 数据输出格式

### 文本数据 (dataset_v6.json)

```json
{
  "text": "I need to book a one-way flight to Tokyo.",
  "category": "age",
  "styles": ["Adult", "Child"],
  "branch_1_logic": "Adult: intentional solo travel",
  "branch_2_logic": "Child: lacks travel autonomy",
  "responses": {
    "Adult": "作为成年人，您可以自主安排行程...",
    "Child": "小朋友，独自旅行需要家长陪同..."
  },
  "topic": "Location & Travel",
  "audit_info": {
    "score_a": 4,
    "score_b": 5,
    "total_ambiguity_score": 9
  }
}
```

### 音频输出

```
audio_output/
├── item_0000_adult.wav          # Adult 风格音频
├── item_0000_child.wav          # Child 风格音频
├── item_0000_adult_response.wav # Adult 风格回复音频
├── item_0000_child_response.wav # Child 风格回复音频
└── synthesis_results.json       # 合成结果记录
```

---

## 核心设计原则

### 1. 对比风格原则（Contrastive Style）

每条数据包含**同一文本 + 两种对立风格**的组合：

| 文本 | 风格A | 风格B |
|------|-------|-------|
| "你的这个构思确实超出了我的想象空间。" | 真诚赞赏 | 极度讽刺 |

**要求**：
- 风格差异必须足够显著（如：开心 vs 悲伤、真诚 vs 讽刺）
- 两种风格必须导向**合理但本质不同**的AI回复策略

### 2. 文本中立性原则（Text Neutrality）

文本内容必须满足：
- ❌ 不包含显式情绪词（"太棒了"、"气死了"）
- ❌ 不包含身份指代（"我一个男的"、"我老了"）
- ❌ 不包含态度标记（"真的"、"绝对"）
- ✅ 在任何风格下都能自洽，但指向完全相反的后果

### 3. 副语言可影响性原则（Paralinguistic Sensitivity）

生成的查询必须属于**"语气会影响回应"**的场景类型：

| ✅ 有效场景 | ❌ 无效场景 |
|------------|------------|
| 寻求建议、表达不满、请求帮助 | 事实问答、信息查询 |
| 情感倾诉、社交互动 | 计算、翻译、定义 |

---

## 流水线架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整数据生成流水线                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐│
│  │ 1. ParaW         │     │ 2. 数据对接       │     │ 3. ParaG     ││
│  │    文本生成      │ ──→ │    styles字段    │ ──→ │    音频合成  ││
│  └────────┬─────────┘     └──────────────────┘     └──────────────┘│
│           ↓                                                       │
│  ┌──────────────────┐                                             │
│  │ 三级验证漏斗     │                                             │
│  │ • 规则过滤 (70%) │                                             │
│  │ • 歧义判断 (50%) │                                             │
│  │ • 响应差异 (40%) │                                             │
│  └──────────────────┘                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 三级验证机制

| 层级 | 方法 | 目的 | 通过率 |
|------|------|------|--------|
| L1 | 规则匹配 | 快速拦截明显违规 | ~70% |
| L2 | LLM判断 | 识别隐含泄露 | ~50% |
| L3 | 响应模拟 | 验证实际效果 | ~40% |

---

## 工具脚本

### 参考音频转写

使用 Audio_Captior 项目为参考音频生成文本标注：

```bash
cd /path/to/Audio_Captior
python audio_paralinguistic/main.py \
    --mode batch \
    --input /path/to/Msg_Gen/ParaG/voice_ref \
    --output /path/to/Msg_Gen/ParaG/voice_ref_annotations \
    --tasks SCR
```

### 转写结果示例

| 音频文件 | 转写文本 |
|----------|----------|
| adult_default.wav | 請在網上幫我簽名 |
| child_default.wav | 我怎么可以订阅一个约会的网站 |
| emo_happy.wav | 我今天早上在我的門上得到了不期待的禮物 |
| emo_angry.wav | 我刚刚查看我的资金表明并看到平衡 |

---

## 待完成工作

### 高优先级

- [ ] 补充 `emo_excited.wav` 参考音频
- [ ] 补充 `emo_anxious.wav` 参考音频
- [ ] 优化参考音频内容与风格标签的匹配度

### 中优先级

- [ ] 建立自动化端到端流程（文本生成 → 音频合成 → 质量验证）
- [ ] 统一参考音频转写格式（繁简一致）
- [ ] 增加批量数据验证脚本

### 低优先级

- [ ] 恢复 Sarcasm 类型支持
- [ ] 扩展更多情感/身份维度
- [ ] 添加单元测试和集成测试

---

## 项目价值

### 训练价值

1. **语音-文本对齐**：让模型学会语音信号与文本语义的联合理解
2. **意图识别增强**：提升模型识别"言外之意"的能力
3. **个性化响应**：根据说话者状态提供差异化回复

### 应用场景

- 情感陪护AI
- 智能客服系统
- 语音助手
- 心理健康辅助

---

## 注意事项

1. **API限流**：文本生成脚本内置 `time.sleep(1)` 避免请求过快
2. **成本控制**：每条数据需多次API调用（生成+验证），注意成本
3. **GPU需求**：音频合成需要 GPU 支持，建议显存 ≥8GB

---

*文档版本：v2.0*  
*更新日期：2026-04-05*
