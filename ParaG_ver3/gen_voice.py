import importlib.util
from pathlib import Path
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, GenerationConfig
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

pretrained_model_name_or_path = "/home/u2023112559/qix/Models/Models/MOSS-TTS-Local"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = """亲爱的你，
你好呀。

今天，我想用最认真、最温柔的声音，对你说一些重要的话。
这些话，像一颗小小的星星，希望能在你的心里慢慢发光。

首先，我想祝你——
每天都能平平安安、快快乐乐。

希望你早上醒来的时候，
窗外有光，屋子里很安静，
你的心是轻轻的，没有着急，也没有害怕。
"""

ref_text_1 = "太阳系八大行星之一。"
# Use audio from ./assets/audio to avoid downloading from the cloud.
ref_audio_1 = "/home/u2023112559/qix/Project/Graduation_Project/SFT_DATA/Msg_Gen/ParaG_ver3/MOSS-TTS/assets/audio/reference_zh.wav"

conversations = [
    # Continuatoin only
    [
        processor.build_user_message(text=ref_text_1 + text_1),
        processor.build_assistant_message(audio_codes_list=[ref_audio_1])
    ],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

batch_size = 1

save_dir = Path("inference_root_moss_tts_local_transformer_continuation")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="continuation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)
