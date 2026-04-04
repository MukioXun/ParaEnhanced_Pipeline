"""
Audio Generation Module for TTS Synthesis
读取 dataset JSON 文件，根据 style 标签选择参考音色进行 TTS 合成
"""
import sys
sys.path.append('CosyVoice/third_party/Matcha-TTS')
from CosyVocie.cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ================= 音色字典配置 =================
# 参考音频文件路径配置（后续补充实际音频文件）
VOICE_REFERENCE_DIR = Path("./voice_ref")  # 参考音频目录
MODEL_PATH = Path("/home/u2023112559/qix/Models/Models/CosyVoice3")  # TTS 模型路径

@dataclass
class VoiceReference:
    """音色参考配置"""
    style: str           # 风格标签
    audio_path: str      # 参考音频路径
    description: str     # 音色描述

# 音色字典：style -> 参考音频配置
VOICE_DICTIONARY: Dict[str, VoiceReference] = {
    # Age 相关音色
    "Adult": VoiceReference(
        style="Adult",
        audio_path=str(VOICE_REFERENCE_DIR / "adult_default.wav"),
        description="成熟稳重的成年男声/女声"
    ),
    "Child": VoiceReference(
        style="Child",
        audio_path=str(VOICE_REFERENCE_DIR / "child_default.wav"),
        description="稚嫩活泼的儿童声音"
    ),

    # Gender 相关音色
    "Male": VoiceReference(
        style="Male",
        audio_path=str(VOICE_REFERENCE_DIR / "male_default.wav"),
        description="男性声音"
    ),
    "Female": VoiceReference(
        style="Female",
        audio_path=str(VOICE_REFERENCE_DIR / "female_default.wav"),
        description="女性声音"
    ),

    # Emotion 相关音色
    "Happy": VoiceReference(
        style="Happy",
        audio_path=str(VOICE_REFERENCE_DIR / "emo_happy.wav"),
        description="开心愉悦的情绪"
    ),
    "Sad": VoiceReference(
        style="Sad",
        audio_path=str(VOICE_REFERENCE_DIR / "emo_sad.wav"),
        description="悲伤低沉的情绪"
    ),
    "Excited": VoiceReference(
        style="Excited",
        audio_path=str(VOICE_REFERENCE_DIR / "emo_excited.wav"),
        description="兴奋激动的情绪"
    ),
    "Anxious": VoiceReference(
        style="Anxious",
        audio_path=str(VOICE_REFERENCE_DIR / "emo_anxious.wav"),
        description="焦虑不安的情绪"
    ),
    "Surprised": VoiceReference(
        style="Surprised",
        audio_path=str(VOICE_REFERENCE_DIR / "emo_surprised.wav"),
        description="惊讶吃惊的情绪"
    ),
    "Disgust": VoiceReference(
        style="Disgust",
        audio_path=str(VOICE_REFERENCE_DIR / "emo_disgust.wav"),
        description="厌恶反感的情绪"
    ),
    "Angry": VoiceReference(
        style="Angry",
        audio_path=str(VOICE_REFERENCE_DIR / "emo_angry.wav"),
        description="愤怒生气的情绪"
    ),
}

# 风格分类映射
STYLE_CATEGORIES = {
    "age": ["Adult", "Child"],
    "gender": ["Male", "Female"],
    "emotion": ["Happy", "Sad", "Excited", "Anxious", "Surprised", "Disgust", "Angry"]
}

# ================= 核心类 =================

class AudioGenerator:
    """音频生成器：根据 style 标签合成 TTS 音频"""

    def __init__(self, voice_dict: Dict[str, VoiceReference] = None):
        """
        初始化音频生成器

        Args:
            voice_dict: 自定义音色字典，默认使用 VOICE_DICTIONARY
        """
        self.voice_dict = voice_dict or VOICE_DICTIONARY
        self._tts_model = None  # 延迟加载 TTS 模型

    def _load_tts_model(self):
        """延迟加载 TTS 模型（如 CosyVoice）"""
        if self._tts_model is None:
            # TODO: 实际加载模型
            # from CosyVoice.cosyvoice.cli.cosyvoice import AutoModel
            self._tts_model = AutoModel(model_dir=MODEL_PATH)
            print("[INFO] TTS Model loading... (placeholder)")
            self._tts_model = "placeholder_model"
        return self._tts_model

    def get_voice_reference(self, style: str) -> Optional[VoiceReference]:
        """
        根据 style 获取音色参考配置

        Args:
            style: 风格标签 (如 "Adult", "Happy", "Male")

        Returns:
            VoiceReference 或 None
        """
        return self.voice_dict.get(style)

    def synthesize(
        self,
        text: str,
        style: str,
        output_path: str,
        use_zero_shot: bool = True
    ) -> bool:
        """
        合成音频

        Args:
            text: 待合成的文本
            style: 风格标签
            output_path: 输出音频路径
            use_zero_shot: 是否使用 zero-shot TTS

        Returns:
            bool: 是否成功
        """
        voice_ref = self.get_voice_reference(style)
        if voice_ref is None:
            print(f"[ERROR] Unknown style: {style}")
            return False

        # 检查参考音频是否存在
        if not os.path.exists(voice_ref.audio_path):
            print(f"[WARN] Reference audio not found: {voice_ref.audio_path}")
            print(f"[INFO] Using placeholder synthesis for style: {style}")

        # 实际 TTS 合成逻辑
        try:
            model = self._load_tts_model()

            if use_zero_shot and os.path.exists(voice_ref.audio_path):
                # Zero-shot TTS：使用参考音频
                for i, result in enumerate(model.inference_zero_shot(
                    text,
                    f"You are a helpful assistant.<|endofprompt|>{voice_ref.description}",
                    voice_ref.audio_path,
                    stream=False
                )):
                    torchaudio.save(output_path, result['tts_speech'], model.sample_rate)
                print(f"[TTS] Synthesizing: '{text[:30]}...' with style '{style}'")
                print(f"[TTS] Reference: {voice_ref.audio_path}")
                print(f"[TTS] Output: {output_path}")
            else:
                # 常规 TTS
                print(f"[TTS] Regular synthesis for style: {style}")

            return True

        except Exception as e:
            print(f"[ERROR] Synthesis failed: {e}")
            return False

    def process_dataset(
        self,
        input_json: str,
        output_dir: str,
        synthesize_responses: bool = True
    ) -> List[Dict]:
        """
        处理整个数据集

        Args:
            input_json: 输入 JSON 文件路径 (dataset_v6.json 格式)
            output_dir: 输出音频目录
            synthesize_responses: 是否也合成 response 部分

        Returns:
            处理结果列表
        """
        # 读取数据
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        os.makedirs(output_dir, exist_ok=True)
        results = []

        for idx, item in enumerate(data):
            text = item.get("text", "")
            styles = item.get("styles", [])
            category = item.get("category", "")
            responses = item.get("responses", {})

            item_result = {
                "index": idx,
                "text": text,
                "category": category,
                "styles": styles,
                "audio_files": {}
            }

            print(f"\n[{idx+1}/{len(data)}] Processing: {text[:50]}...")

            # 为每个 style 合成音频
            for style in styles:
                # 合成主文本
                audio_path = os.path.join(output_dir, f"item_{idx:04d}_{style.lower()}.wav")
                success = self.synthesize(text, style, audio_path)
                item_result["audio_files"][style] = {
                    "main_text": audio_path if success else None
                }

                # 合成对应的 response（可选）
                if synthesize_responses and style in responses:
                    response_text = responses[style]
                    resp_audio_path = os.path.join(
                        output_dir,
                        f"item_{idx:04d}_{style.lower()}_response.wav"
                    )
                    resp_success = self.synthesize(response_text, style, resp_audio_path)
                    item_result["audio_files"][style]["response"] = resp_audio_path if resp_success else None

            results.append(item_result)

        # 保存处理结果
        result_path = os.path.join(output_dir, "synthesis_results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n[DONE] Processed {len(results)} items. Results saved to: {result_path}")
        return results


# ================= 便捷函数 =================

def get_style_info(style: str) -> Optional[Dict]:
    """获取风格标签的详细信息"""
    voice = VOICE_DICTIONARY.get(style)
    if voice:
        return {
            "style": voice.style,
            "audio_path": voice.audio_path,
            "description": voice.description
        }
    return None

def list_all_styles() -> Dict[str, List[str]]:
    """列出所有可用的风格标签"""
    return STYLE_CATEGORIES.copy()

def validate_voice_references() -> Dict[str, bool]:
    """检查所有参考音频文件是否存在"""
    status = {}
    for style, voice in VOICE_DICTIONARY.items():
        status[style] = os.path.exists(voice.audio_path)
    return status


# ================= 命令行入口 =================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Generation for TTS")
    parser.add_argument("--input", "-i", default="results/dataset_v6.json",
                        help="Input JSON file path")
    parser.add_argument("--output", "-o", default="audio_output",
                        help="Output directory for generated audio")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Validate voice reference files")
    parser.add_argument("--list-styles", "-l", action="store_true",
                        help="List all available styles")

    args = parser.parse_args()

    if args.list_styles:
        print("Available Styles:")
        for category, styles in list_all_styles().items():
            print(f"  {category}: {', '.join(styles)}")

    elif args.validate:
        print("Validating voice references:")
        status = validate_voice_references()
        for style, exists in status.items():
            mark = "✓" if exists else "✗"
            print(f"  [{mark}] {style}: {VOICE_DICTIONARY[style].audio_path}")

    else:
        generator = AudioGenerator()
        generator.process_dataset(args.input, args.output)
