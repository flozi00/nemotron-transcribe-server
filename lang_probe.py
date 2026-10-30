import base64, json, pathlib, urllib.request

HOST = "https://parakeet-primeline-v3.pl-ai.net"
a = base64.b64encode(pathlib.Path("/tmp/t.wav").read_bytes()).decode()

PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including "
    "fillers and repetitions. Formatting: punctuation and capitalization. "
    "Output: the ISO 639-1 code of the spoken language in square brackets, "
    "then the transcript only."
)

body = {
    "model": "primeline/parakeet-primeline",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
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
print("content:", repr(d["choices"][0]["message"]["content"]))
print("usage:", d["usage"])
