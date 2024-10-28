import requests
res = requests.post(f"http://127.0.0.1:5000/asr", files={"file": open("c:/users/c1/videos/5s.wav", 'rb')},data={"lang":"zh"}, timeout=7200)
print(res.json())