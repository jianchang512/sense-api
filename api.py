# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from pathlib import Path
import time
from datetime import timedelta
from funasr import AutoModel
import torch
import shutil
from pydub import AudioSegment

TMPDIR=Path(os.path.dirname(__file__)+"/tmp").as_posix()
Path(TMPDIR).mkdir(exist_ok=True)
device="cuda:0" if torch.cuda.is_available() else "cpu"

HOST='127.0.0.1'
PORT=5000


'''
格式化毫秒或秒为符合srt格式的 2位小时:2位分:2位秒,3位毫秒 形式
print(ms_to_time_string(ms=12030))
-> 00:00:12,030
'''
def ms_to_time_string(*, ms=0, seconds=None):
    # 计算小时、分钟、秒和毫秒
    if seconds is None:
        td = timedelta(milliseconds=ms)
    else:
        td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000

    time_string = f"{hours}:{minutes}:{seconds},{milliseconds}"
    return format_time(time_string, ',')

# 将不规范的 时:分:秒,|.毫秒格式为  aa:bb:cc,ddd形式
# eg  001:01:2,4500  01:54,14 等做处理
def format_time(s_time="", separate=','):
    if not s_time.strip():
        return f'00:00:00{separate}000'
    hou, min, sec,ms = 0, 0, 0,0

    tmp = s_time.strip().split(':')
    if len(tmp) >= 3:
        hou,min,sec = tmp[-3].strip(),tmp[-2].strip(),tmp[-1].strip()
    elif len(tmp) == 2:
        min,sec = tmp[0].strip(),tmp[1].strip()
    elif len(tmp) == 1:
        sec = tmp[0].strip()
    
    if re.search(r',|\.', str(sec)):
        t = re.split(r',|\.', str(sec))
        sec = t[0].strip()
        ms=t[1].strip()
    else:
        ms = 0
    hou = f'{int(hou):02}'[-2:]
    min = f'{int(min):02}'[-2:]
    sec = f'{int(sec):02}'
    ms = f'{int(ms):03}'[-3:]
    return f"{hou}:{min}:{sec}{separate}{ms}"

def remove_unwanted_characters(text: str) -> str:
    # 保留中文、日文、韩文、英文、数字和常见符号，去除其他字符
    allowed_characters = re.compile(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af'
                                    r'a-zA-Z0-9\s.,!@#$%^&*()_+\-=\[\]{};\'"\\|<>/?，。！｛｝【】；‘’“”《》、（）￥]+')
    return re.sub(allowed_characters, '', text)


model = AutoModel(
    model="iic/SenseVoiceSmall",
    punc_model="ct-punc", 
    disable_update=True,
    device=device
)
#vad
vm = AutoModel(model="fsmn-vad",max_single_segment_time=20000,max_end_silence_time=250,disable_update=True,device=device)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            api 地址为 http://{HOST}:{PORT}/asr
        </body>
    </html>
    """

@app.post("/asr")
async def asr(file: UploadFile, lang: str = Form(...)):
    print(f'{lang=},{file.filename=}')
    if lang not in ['zh','ja','en','ko','yue']:
        return {"code":1,"msg":f'不支持的语言代码:{lang}'}
    # 创建一个临时文件路径
    temp_file_path = f"{TMPDIR}/{file.filename}"
    ## 将上传的文件保存到临时路径
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    segments = vm.generate(input=temp_file_path)
    audiodata = AudioSegment.from_file(temp_file_path)    
    
    srts=[]
    for seg in segments[0]['value']:
        chunk=audiodata[seg[0]:seg[1]]
        filename=f"{TMPDIR}/{seg[0]}-{seg[1]}.wav"
        chunk.export(filename)
        res = model.generate(
            input=filename,
            language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True
        )
        text = remove_unwanted_characters(rich_transcription_postprocess(res[0]["text"]))
        print(f'{text=}')
        srts.append(f'{len(srts)+1}\n{ms_to_time_string(ms=seg[0])} --> {ms_to_time_string(ms=seg[1])}\n{text.strip()}')
    return {"code":0,"msg":"ok","data":"\n\n".join(srts)}


if __name__=='__main__':
    import uvicorn
    uvicorn.run("api:app", host=HOST,port=PORT, log_level="info")
