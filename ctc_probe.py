import base64, json, pathlib, sys, urllib.request

HOST = "https://parakeet-primeline-v3.pl-ai.net"
a = base64.b64encode(pathlib.Path("/tmp/t.wav").read_bytes()).decode()
body = {
    "model": "primeline/parakeet-primeline",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe the following audio."},
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + a}},
    ]}],
    "temperature": 0, "max_tokens": 2048,
    "skip_special_tokens": False, "spaces_between_special_tokens": False, "seed": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "mm_processor_kwargs": {"capture_ctc_timestamps": True},
}
req = urllib.request.Request(HOST + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.loads(r.read())
print("top-level keys:", sorted(d))
print("content:", repr(d["choices"][0]["message"]["content"]))
ctc = d.get("ctc_timestamps")
print("ctc type:", type(ctc).__name__)
if isinstance(ctc, dict):
    print("ctc keys:", sorted(ctc))
    for k, v in ctc.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} items, first 3 = {v[:3]}")
        else:
            print(f"  {k}: {v!r}"[:300])
