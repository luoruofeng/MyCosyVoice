import sys
import os
sys.path.append('third_party\\Matcha-TTS')
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import concurrent.futures

# 加载模型
model_dir = 'pretrained_models/CosyVoice2-0.5B'  # 调整为您的模型目录
cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 参考文件路径
reference_audio_path = 'my_voice/girl.mp3'
reference_text_path = 'my_voice/girl.txt'

# 读取参考文本
with open(reference_text_path, 'r', encoding='utf-8') as f:
    prompt_text = f.read().strip()

# 加载参考音频并重采样到 16kHz
prompt_speech_16k = load_wav(reference_audio_path, 16000)

# 添加 zero_shot_spk
zero_shot_spk_id = 'my_girl_spk'
cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)
cosyvoice.save_spkinfo()

def generate_audio(tts_text, output_path):
    audio_segments = []
    for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=zero_shot_spk_id, stream=False)):
        audio_segments.append(j['tts_speech'])
    if audio_segments:
        full_audio = torch.cat(audio_segments, dim=-1)
        torchaudio.save(output_path, full_audio, cosyvoice.sample_rate, format='mp3')

def process_file(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        tts_text = f.read().strip()
    output_dir = os.path.dirname(txt_path)
    output_filename = os.path.splitext(os.path.basename(txt_path))[0] + '.mp3'
    output_path = os.path.join(output_dir, output_filename)
    generate_audio(tts_text, output_path)
    print(f'Generated: {output_path}')

def process_directory(dir_path):
    txt_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    # 顺序处理，避免并发问题
    for txt_path in txt_files:
        try:
            process_file(txt_path)
        except Exception as e:
            print(f'Error processing file {txt_path}: {e}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python run.py <txt_path_or_dir>')
        sys.exit(1)
    path = sys.argv[1]
    if os.path.isfile(path) and path.lower().endswith('.txt'):
        process_file(path)
    elif os.path.isdir(path):
        process_directory(path)
    else:
        print('Invalid path: must be a TXT file or directory')
        sys.exit(1)