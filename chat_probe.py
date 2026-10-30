import base64, json, pathlib, sys, time, urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "https://parakeet-primeline-v3.pl-ai.net"
WAV = sys.argv[2] if len(sys.argv) > 2 else "/tmp/t.wav"
a = base64.b64encode(pathlib.Path(WAV).read_bytes()).decode()
body = {
    "model": "primeline/parakeet-primeline",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe the following audio."},
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + a}},
    ]}],
    "temperature": 0, "top_p": 1.0, "max_tokens": 2048,
    "skip_special_tokens": False, "spaces_between_special_tokens": False, "seed": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
raw = json.dumps(body).encode()
print("request bytes", len(raw))
req = urllib.request.Request(HOST + "/v1/chat/completions", data=raw,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        out = r.read().decode()
        print("code", r.status, "time %.1fs" % (time.time() - t0))
        print(out[:2000])
except Exception as exc:
    print("FAILED after %.1fs: %r" % (time.time() - t0, exc))
    body = getattr(exc, "read", lambda: b"")()
    if body:
        print(body.decode()[:2000])
