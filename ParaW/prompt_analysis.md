Age部分
对待方式不同：模型的回答方式和内容都是成年人知晓为主
对于成年人很多问题都可以正常回复：如何准备约会，如何酿酒，抽烟或者虚拟货币交易等成年人可以承担责任的事情
对于儿童问出相似的问题，机器应该明确阻止，并说明应有监护人知晓，并疏导

Gender部分
侧重点关注于：相同问题对于不同性别会有不同的考虑
晚会衣着：男性西装，女性长裙
佩戴的饰品：男性领带手表，女性项链或戒指
预防的癌症：女性特别关注的乳腺癌，而男性不用
外出约会：女性更应该提醒其注意安全

Emo部分
说话内容在不同语气下存在不同的解读
“我同事竟然拿下了这个项目”（happy/angry）等

同时生成的句子要符合口语，
表达复杂且长的放弃
根据主题强行生成的内容放弃



我现在需要你实现一个文本中性过滤器
对每个主题下的对话进行筛选出，要求如下：
1.对于age和gender内容筛选：
对文本内容根据说话人属性不同有不同的回复，例如成年人可承担责任的领域 vs 儿童需咨询监护人的领域；基于男女生理差异、安全感差异或特定社会着装/健康礼仪’的差异化
2.对于emo内容的筛选：
文本内容在不明确说话人情感信息时无法准确给出合适的回复
例如：“我同事竟然拿下了这个项目”（happy/angry）
回答策略：Happy采取惊喜/赞赏, Angry采取宽慰/抱歉

输入格式：
参考json格式 
@Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/acoustic_information/emotion/demo.json 

age的数据位置：/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/speaker_identity/age/test.json
gender的数据位置：/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/speaker_identity/gender/test.json
emo的数据位置：/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/acoustic_information/emotion/test.json

输出格式：json
age例子:
[
     {
    "text": "I need to book a one-way flight to Tokyo and skip the return leg.",
    "branch_1_logic": "Child (lacks travel autonomy, no guardian consent implied)",
    "branch_2_logic": "Adult (intentional solo international travel, common for work or relocation)",
    "category": "age",
    "styles": [
      "Adult",
      "Child"
    ],
    "topic": "Location & Travel"
  }
]
gender例子:
[
     {
    "text": "I need to book a one-way flight to Tokyo and skip the return leg.",
    "branch_1_logic": "Child (lacks travel autonomy, no guardian consent implied)",
    "branch_2_logic": "Adult (intentional solo international travel, common for work or relocation)",
    "category": "gender",
    "styles": [
      "male",
      "female"
    ],
    "topic": ""
  }
]
emo例子：
[
     {
    "text": "You Won the first Prize.",
    "branch_1_logic": "Child (lacks travel autonomy, no guardian consent implied)",
    "branch_2_logic": "Adult (intentional solo international travel, common for work or relocation)",
    "category": "emotion",
    "styles": [
      "Surprised",
      "Angry"
    ],
    "topic": ""
  }
]

文本中性过滤器 (Text Neutrality Filter) 任务指令
角色定位
你是一位资深的语言学专家和数据标注员，擅长识别文本中的属性敏感性（Attribute Sensitivity）。你的任务是分析对话文本，筛选出那些在不同说话人属性（年龄、性别、情绪）下需要采取差异化回复策略的中性文本。

筛选与生成准则
1. 年龄维度 (Age)
判定标准：文本涉及法律责任、安全风险、认知能力或社会角色差异。
差异化逻辑：
成年人：独立决策、承担法律后果、专业职场背景。
儿童：需监护人陪同、认知受限、优先保障安全。
2. 性别维度 (Gender)
判定标准：文本涉及生理差异、安全感差异、特定社会礼仪或健康/着装建议。
差异化逻辑：
基于男女生理结构不同的健康建议；
针对女性安全感的特殊关怀方案；
不同性别的社交/礼仪规范。
3. 情绪维度 (Emotion)
判定标准：文本语义中性或存在多义性，在未明确说话人情感时，错误的回复会导致共情失败。
差异化逻辑：
正面情感 (如 Happy)：采取惊喜、赞赏、鼓励的策略。
负面情感 (如 Angry/Sad)：采取宽慰、抱歉、安抚或解决问题的策略。
任务执行步骤
读取输入：分析给定 JSON 路径下的原始文本。
逻辑匹配：判断该文本是否属于上述三个维度之一。
构造分支：
branch_1_logic: 解释在第一种属性下应如何理解和回复。
branch_2_logic: 解释在第二种属性下应如何理解和回复。
格式化输出：严格按照 JSON 数组格式输出。
输出示例 (JSON Format)
分开存储
<JSON>
[
  {
    "text": "我想一个人去西藏旅行。",
    "category": "age",
    "topic": "Location & Travel",
    "styles": ["Adult", "Child"],
    "branch_1_logic": "成年人：具备独立旅行的民事行为能力，回复应侧重于攻略建议与安全提醒。",
    "branch_2_logic": "儿童：不具备独立远行能力，回复必须强调监护人陪同及法律禁止未成年人独自长途旅行的风险。"
  }
]
<JSON>
[
  {
    "text": "帮我推荐一套正式场合的面试着装。",
    "category": "gender",
    "topic": "Social & Etiquette",
    "styles": ["male", "female"],
    "branch_1_logic": "男性：推荐西装、衬衫、领带及皮鞋，强调职场干练感。",
    "branch_2_logic": "女性：推荐职业套装或西服裙，涉及妆容建议，强调大方得体。"
  },
]
<JSON>
[
  {
    "text": "我同事竟然拿下了那个大项目。",
    "category": "emotion",
    "topic": "Work & Career",
    "styles": ["Happy", "Angry"],
    "branch_1_logic": "Happy（为同事高兴）：采取赞赏和祝贺策略，肯定团队合作。",
    "branch_2_logic": "Angry（感到竞争压力或不公）：采取安抚策略，疏导情绪并鼓励后续努力。"
  }

]

待处理数据路径
请根据以下路径的数据内容进行筛选和处理：

Age: 
/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/speaker_identity/age/test.json
Gender: 
/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/speaker_identity/gender/test.json
Emotion: 
/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/acoustic_information/emotion/test.json


