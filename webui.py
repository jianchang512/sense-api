# coding=utf-8

import os,re,time,threading
import librosa
import base64
import io
import gradio as gr

import numpy as np
import torch
import torchaudio

from funasr import AutoModel

model = AutoModel(model="iic/SenseVoiceSmall",
                  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  vad_kwargs={"max_single_segment_time": 30000},
                  trust_remote_code=True,
                  )



emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def open_page():
    import webbrowser
    time.sleep(5)
    webbrowser.open_new_tab(f'http://127.0.0.1:7860')
    
def format_str(s):
    for sptk in emoji_dict:
        s = s.replace(sptk, emoji_dict[sptk])
    return s


def format_str_v2(s):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s = s + emo_dict[emo]

    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()

def format_str_v3(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None
    def get_event(s):
        return s[0] if s[0] in event_set else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        #else:
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()

def model_inference(input_wav, language, fs=16000):
    # task_abbr = {"Speech Recognition": "ASR", "Rich Text Transcription": ("ASR", "AED", "SER")}
    language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
                     "nospeech": "nospeech"}
    
    # task = "Speech Recognition" if task is None else task
    language = "auto" if len(language) < 1 else language
    selected_language = language_abbr[language]
    # selected_task = task_abbr.get(task)
    
    # print(f"input_wav: {type(input_wav)}, {input_wav[1].shape}, {input_wav}")
    
    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            print(f"audio_fs: {fs}")
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    
    
    merge_vad = True #False if selected_task == "ASR" else True
    print(f"language: {language}, merge_vad: {merge_vad}")
    text = model.generate(input=input_wav,
                          cache={},
                          language=language,
                          use_itn=True,
                          batch_size_s=60, merge_vad=merge_vad)
    
    text = text[0]["text"]
    text = format_str_v3(text)
    
    print(text)
    
    return text


html_content = """

<div>
    <h2 style="font-size: 22px;margin-left: 0px;">SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small 是一种纯编码器语音基础模型，专为快速语音理解而设计</p>
    <p style="margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice阿里官方GitHub</a>
    <a href="https://github.com/jianchang512/sense-api" target="_blank">Sense-Api仓库</a> 
    <a href="https://github.com/jianchang512/pyvideotrans" target="_blank">pyVideoTrans仓库</a></p>
</div>
"""


def launch():
    with gr.Blocks(theme=gr.themes.Soft(),title="SenseVoice 在线web界面") as demo:
        #gr.Interface()
        # gr.Markdown(description)
        gr.HTML(html_content)
        with gr.Row():
            with gr.Column():
                audio_inputs = gr.Audio(label="上传音频或录制麦克风")
                
                with gr.Accordion("配置"):
                    language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                                                  value="auto",
                                                  label="说话语言")
                fn_button = gr.Button("开始识别", variant="primary")
                text_outputs = gr.Textbox(label="识别结果")
            #gr.Examples(examples=audio_examples, inputs=[audio_inputs, language_inputs], examples_per_page=20)
        
        fn_button.click(model_inference, inputs=[audio_inputs, language_inputs], outputs=text_outputs)
    threading.Thread(target=open_page).start()
    demo.launch()


if __name__ == "__main__":
    # iface.launch()   
    launch()


