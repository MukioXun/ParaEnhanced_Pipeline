#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本中性过滤器 (Text Neutrality Filter)
使用 Qwen API 分析对话文本，筛选出需要差异化回复策略的中性文本

特性:
- 流式保存: 处理过程中实时保存结果
- 断点重连: 支持从中断处继续处理
"""

import json
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Set
from openai import OpenAI
from tqdm import tqdm
import argparse


class TextNeutralityFilter:
    """文本中性过滤器类

    支持流式保存和断点重连
    """

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化过滤器

        Args:
            api_key: Qwen API 密钥
            base_url: API 基础 URL
            checkpoint_dir: 检查点目录，用于保存进度
        """
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen-plus"  # 可根据需要修改为 qwen-max 等
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _get_text_hash(self, text: str) -> str:
        """生成文本的唯一标识"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]

    def _get_checkpoint_path(self, category: str) -> str:
        """获取检查点文件路径"""
        return os.path.join(self.checkpoint_dir, f"{category}_checkpoint.json")

    def _get_stats_path(self, category: str) -> str:
        """获取统计信息文件路径"""
        return os.path.join(self.checkpoint_dir, f"{category}_stats.json")

    def _get_results_path(self, category: str) -> str:
        """获取结果文件路径"""
        return os.path.join(self.checkpoint_dir, f"{category}_results.json")

    def _load_checkpoint(self, category: str) -> Dict:
        """加载检查点数据"""
        checkpoint_path = self._get_checkpoint_path(category)
        results_path = self._get_results_path(category)
        stats_path = self._get_stats_path(category)

        checkpoint = {
            "processed_hashes": set(),
            "results": [],
            "stats": {
                "total_processed": 0,
                "first_filter_passed": 0,
                "second_filter_passed": 0
            }
        }

        # 加载已处理的文本哈希
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    checkpoint["processed_hashes"] = set(data.get("processed_hashes", []))
                print(f"加载检查点: 已处理 {len(checkpoint['processed_hashes'])} 条文本")
            except Exception as e:
                print(f"加载检查点失败: {e}")

        # 加载已有结果
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    checkpoint["results"] = json.load(f)
                print(f"加载已有结果: {len(checkpoint['results'])} 条")
            except Exception as e:
                print(f"加载结果失败: {e}")

        # 加载统计信息
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    checkpoint["stats"] = json.load(f)
            except Exception as e:
                print(f"加载统计信息失败: {e}")

        return checkpoint

    def _save_checkpoint(self, category: str, processed_hashes: Set[str]):
        """保存检查点（已处理的文本哈希）"""
        checkpoint_path = self._get_checkpoint_path(category)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump({"processed_hashes": list(processed_hashes)}, f, ensure_ascii=False)

    def _save_stats(self, category: str, stats: Dict):
        """保存统计信息"""
        stats_path = self._get_stats_path(category)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def _save_results_incremental(self, category: str, results: List[Dict]):
        """增量保存结果"""
        results_path = self._get_results_path(category)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def load_json_data(self, file_path: str) -> Dict:
        """加载 JSON 数据文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_texts_from_dialog(self, dialog_data: Dict, category: str) -> List[Dict]:
        """
        从对话数据中提取文本

        Args:
            dialog_data: 对话数据字典
            category: 类别 (age/gender/emotion)

        Returns:
            包含文本和元数据的列表
        """
        texts = []
        for topic, dialogs in dialog_data.items():
            for dialog_id, turns in dialogs.items():
                if not turns:  # 跳过空对话
                    continue
                for turn in turns:
                    text_info = {
                        "text": turn.get("content", ""),
                        "topic": topic,
                        "category": category,
                        "insturct_style": turn.get("insturct_style", ""),
                        "spk_info": turn.get("spk_info", ""),
                        "dialog_id": dialog_id,
                        "turn_id": turn.get("turn_id", 0)
                    }
                    texts.append(text_info)
        return texts

    def build_analysis_prompt(self, text_info: Dict) -> str:
        """构建分析提示词"""
        text = text_info["text"]
        category = text_info["category"]

        category_guidelines = {
            "age": """
【年龄维度分析】
判定标准：文本涉及法律责任、安全风险、认知能力或社会角色差异。
差异化逻辑：
- 成年人：独立决策、承担法律后果、专业职场背景
- 儿童：需监护人陪同、认知受限、优先保障安全
请判断该文本是否需要针对成年人和儿童采取不同的回复策略。""",

            "gender": """
【性别维度分析】
判定标准：文本涉及生理差异、安全感差异、特定社会礼仪或健康/着装建议。
差异化逻辑：
- 基于男女生理结构不同的健康建议
- 针对女性安全感的特殊关怀方案
- 不同性别的社交/礼仪规范
请判断该文本是否需要针对男性和女性采取不同的回复策略。""",

            "emotion": """
【情绪维度分析】
判定标准：文本语义中性或存在多义性，在未明确说话人情感时，错误的回复会导致共情失败。
差异化逻辑：
- 正面情感 (如 Happy)：采取惊喜、赞赏、鼓励的策略
- 负面情感 (如 Angry/Sad)：采取宽慰、抱歉、安抚或解决问题的策略
请判断该文本是否存在情绪多义性，需要针对不同情感采取不同回复策略。"""
        }

        prompt = f"""你是一位资深的语言学专家和数据标注员，擅长识别文本中的属性敏感性（Attribute Sensitivity）。

请分析以下文本，判断它是否属于需要差异化回复的中性文本。

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

注意：
1. 只输出 JSON 格式，不要有其他内容
2. 对于 age 类别，styles 应为 ["Adult", "Child"]
3. 对于 gender 类别，styles 应为 ["male", "female"]
4. 对于 emotion 类别，styles 应根据文本内容选择合适的情绪对，如 ["Happy", "Angry"] 或 ["Happy", "Sad"]
5. branch_logic 应具体说明该属性下应如何理解和回复"""
        return prompt

    def build_naturalness_prompt(self, text: str, category: str, styles: List[str],
                                     branch_logics: List[str]) -> str:
        """
        构建自然度验证提示词

        检验在特定style下，文本表达是否足够自然，不需要过多假设和联想才能解释
        """
        style_pairs = ""
        for i, (style, logic) in enumerate(zip(styles, branch_logics), 1):
            style_pairs += f"\n风格{i}: {style}\n理解方式: {logic}\n"

        prompt = f"""你是一位语言学专家，专门评估文本在特定语境下的自然度和可理解性。

【任务】
检验以下文本在不同风格设定下的表达是否足够自然。

【文本】
"{text}"

【风格设定】
{style_pairs}

【评估标准】
判断该文本在每种风格下的表达是否"自然"：
- 自然：该文本在这种风格设定下，表达直接、清晰，不需要听众做大量假设或联想就能理解说话意图
- 不自然：需要听众进行复杂的推理、假设或联想才能解释为什么这个人会这样说

【示例】
文本: "我明天要去签合同"
- 风格: Adult（成年人）→ 自然（成年人签合同是常见场景）
- 风格: Child（儿童）→ 不自然（儿童通常不会独立签合同，需要大量假设才能解释）

文本: "我想要那个红色的"
- 风格: Happy（开心）→ 自然（开心时表达想要某物很直接）
- 风格: Angry（愤怒）→ 也自然（愤怒时也可以直接表达需求）

请输出 JSON 格式：
{{
    "is_natural": true/false,
    "style_assessments": [
        {{"style": "风格1名称", "is_natural": true/false, "reason": "简短理由"}},
        {{"style": "风格2名称", "is_natural": true/false, "reason": "简短理由"}}
    ],
    "overall_reason": "总结为什么整体自然或不自然"
}}

注意：只有当两种风格下都比较自然时，才认为该文本适合作为中性文本样本。"""
        return prompt

    def verify_naturalness(self, text: str, category: str, styles: List[str],
                           branch_logics: List[str], max_retries: int = 3) -> bool:
        """
        二次筛选：验证文本在各style下的自然度

        Args:
            text: 待验证文本
            category: 类别
            styles: 风格列表
            branch_logics: 各风格的理解逻辑

        Returns:
            是否通过自然度验证
        """
        prompt = self.build_naturalness_prompt(text, category, styles, branch_logics)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位语言学专家，评估文本的自然度和可理解性。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                result_text = response.choices[0].message.content.strip()

                # 移除 markdown 标记
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]

                result = json.loads(result_text.strip())

                # 检查各风格的自然度
                is_natural = result.get("is_natural", False)
                style_assessments = result.get("style_assessments", [])

                # 两种风格都需要自然才通过
                all_natural = all(s.get("is_natural", False) for s in style_assessments)

                return all_natural

            except json.JSONDecodeError as e:
                print(f"自然度验证 JSON 解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"自然度验证 API 错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        return False  # 验证失败时默认不通过

    def analyze_text(self, text_info: Dict, max_retries: int = 3) -> Optional[Dict]:
        """
        使用 Qwen API 分析文本

        Args:
            text_info: 文本信息字典
            max_retries: 最大重试次数

        Returns:
            包含结果和筛选阶段的字典，或 None
            {
                "stage": "first_filter_rejected" | "second_filter_rejected" | "passed",
                "result": {...} | None
            }
        """
        prompt = self.build_analysis_prompt(text_info)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位专业的语言学分析专家，擅长识别文本中的属性敏感性。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                result_text = response.choices[0].message.content.strip()

                # 尝试解析 JSON
                # 移除可能的 markdown 代码块标记
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]

                result = json.loads(result_text.strip())

                if result.get("is_neutral", False):
                    # 初筛通过，进行二次筛选：验证自然度
                    styles = result.get("styles", [])
                    branch_logics = [
                        result.get("branch_1_logic", ""),
                        result.get("branch_2_logic", "")
                    ]

                    is_natural = self.verify_naturalness(
                        text_info["text"],
                        text_info["category"],
                        styles,
                        branch_logics
                    )

                    if is_natural:
                        return {
                            "stage": "passed",
                            "result": {
                                "text": text_info["text"],
                                "category": text_info["category"],
                                "topic": text_info["topic"],
                                "styles": styles,
                                "branch_1_logic": branch_logics[0],
                                "branch_2_logic": branch_logics[1],
                                "reason": result.get("reason", ""),
                                "naturalness_verified": True
                            }
                        }
                    else:
                        # 初筛通过但自然度验证失败
                        return {
                            "stage": "second_filter_rejected",
                            "result": None
                        }
                return {
                    "stage": "first_filter_rejected",
                    "result": None
                }

            except json.JSONDecodeError as e:
                print(f"JSON 解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"API 调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        return None

    def process_category(self, file_path: str, category: str,
                         batch_size: int = 10, max_samples: int = None,
                         save_interval: int = 5) -> List[Dict]:
        """
        处理单个类别的数据（支持断点重连和流式保存）

        Args:
            file_path: JSON 文件路径
            category: 类别名称
            batch_size: 批处理大小
            max_samples: 最大处理样本数（用于测试）
            save_interval: 每处理多少条保存一次

        Returns:
            筛选出的中性文本列表
        """
        print(f"\n正在处理 {category} 类别...")
        print(f"加载文件: {file_path}")

        # 加载检查点
        checkpoint = self._load_checkpoint(category)
        processed_hashes = checkpoint["processed_hashes"]
        neutral_texts = checkpoint["results"]
        stats = checkpoint["stats"]

        data = self.load_json_data(file_path)
        texts = self.extract_texts_from_dialog(data, category)

        # 去重
        unique_texts = {}
        for text_info in texts:
            text = text_info["text"]
            if text and text not in unique_texts:
                unique_texts[text] = text_info

        texts = list(unique_texts.values())
        print(f"共提取 {len(texts)} 条唯一文本")

        # 过滤已处理的文本
        pending_texts = []
        for text_info in texts:
            text_hash = self._get_text_hash(text_info["text"])
            if text_hash not in processed_hashes:
                pending_texts.append(text_info)

        print(f"已处理: {len(processed_hashes)} 条, 待处理: {len(pending_texts)} 条")

        if max_samples:
            pending_texts = pending_texts[:max_samples]
            print(f"限制处理前 {max_samples} 条样本")

        if not pending_texts:
            print("所有文本已处理完成，无需重复处理")
            return neutral_texts

        # 分析文本
        processed_count = 0
        try:
            for text_info in tqdm(pending_texts, desc=f"分析{category}"):
                text_hash = self._get_text_hash(text_info["text"])

                analysis = self.analyze_text(text_info)

                # 更新统计
                stats["total_processed"] += 1
                if analysis:
                    if analysis["stage"] == "passed":
                        stats["first_filter_passed"] += 1
                        stats["second_filter_passed"] += 1
                        neutral_texts.append(analysis["result"])
                    elif analysis["stage"] == "second_filter_rejected":
                        stats["first_filter_passed"] += 1

                # 记录已处理
                processed_hashes.add(text_hash)
                processed_count += 1

                # 流式保存
                if processed_count % save_interval == 0:
                    self._save_checkpoint(category, processed_hashes)
                    self._save_results_incremental(category, neutral_texts)
                    self._save_stats(category, stats)
                    tqdm.write(f"已保存: 已处理 {stats['total_processed']} 条, "
                              f"初筛通过 {stats['first_filter_passed']} 条, "
                              f"最终通过 {stats['second_filter_passed']} 条")

                # 控制 API 调用频率
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n检测到中断，正在保存进度...")
            self._save_checkpoint(category, processed_hashes)
            self._save_results_incremental(category, neutral_texts)
            self._save_stats(category, stats)
            print(f"进度已保存。下次运行将从断点继续。")
            raise
        except Exception as e:
            print(f"\n处理出错: {e}")
            self._save_checkpoint(category, processed_hashes)
            self._save_results_incremental(category, neutral_texts)
            self._save_stats(category, stats)
            print(f"进度已保存。下次运行将从断点继续。")
            raise

        # 最终保存
        self._save_checkpoint(category, processed_hashes)
        self._save_results_incremental(category, neutral_texts)
        self._save_stats(category, stats)

        print(f"\n筛选统计:")
        print(f"  总处理: {stats['total_processed']} 条")
        print(f"  初筛通过: {stats['first_filter_passed']} 条")
        print(f"  自然度验证通过: {stats['second_filter_passed']} 条")
        print(f"  最终筛选率: {stats['second_filter_passed']/max(stats['total_processed'], 1)*100:.1f}%")
        return neutral_texts

    def save_results(self, results: List[Dict], output_path: str):
        """保存结果到 JSON 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

    def save_by_category(self, all_results: Dict[str, List[Dict]], output_dir: str):
        """按类别分开保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        for category, results in all_results.items():
            output_path = os.path.join(output_dir, f"{category}_neutral_texts.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"{category} 类别已保存 {len(results)} 条记录到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="文本中性过滤器（支持流式保存和断点重连）")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="输出目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="检查点目录，用于断点重连")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="每类别最大处理样本数（用于测试）")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="Qwen 模型名称")
    parser.add_argument("--save_interval", type=int, default=20,
                        help="每处理多少条保存一次检查点")
    parser.add_argument("--reset", action="store_true",
                        help="重置检查点，从头开始处理")
    args = parser.parse_args()

    # 数据文件路径
    data_paths = {
        # "age": "/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/speaker_identity/age/test.json",
        # "gender": "/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/speaker_identity/gender/test.json",
        "emotion": "/home/u2023112559/qix/Project/Final_Project/SFT_DATA/Datasets/voxdialogue/JSON/acoustic_information/emotion/test.json"
    }

    # 初始化过滤器
    filter_agent = TextNeutralityFilter(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        checkpoint_dir=args.checkpoint_dir
    )
    filter_agent.model = args.model

    # 重置检查点
    if args.reset:
        import shutil
        if os.path.exists(args.checkpoint_dir):
            shutil.rmtree(args.checkpoint_dir)
            os.makedirs(args.checkpoint_dir)
            print("检查点已重置")

    # 处理所有类别
    all_results = {}
    for category, file_path in data_paths.items():
        if os.path.exists(file_path):
            results = filter_agent.process_category(
                file_path,
                category,
                max_samples=args.max_samples,
                save_interval=args.save_interval
            )
            all_results[category] = results
        else:
            print(f"警告: 文件不存在 - {file_path}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    # 按类别分开保存
    filter_agent.save_by_category(all_results, args.output_dir)

    # 合并保存
    combined_results = []
    for category, results in all_results.items():
        combined_results.extend(results)

    combined_path = os.path.join(args.output_dir, "combined_neutral_texts.json")
    filter_agent.save_results(combined_results, combined_path)

    print(f"\n处理完成! 共筛选出 {len(combined_results)} 条中性文本")


if __name__ == "__main__":
    main()
