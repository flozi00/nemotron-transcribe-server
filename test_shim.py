#!/usr/bin/env python3
"""Local check of the parakeet shim against a fake vLLM upstream.

Runs the real shim process, feeds it a real multipart WAV, and asserts the
response is parakeet-server's schema for both response formats.
"""

import io
import json
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "shim", "parakeet_shim.py")
UPSTREAM_PORT = 18000
SHIM_PORT = 18080

CAPTURED = {}


def wav_bytes(seconds=2.0, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakeVLLM(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "primeline/parakeet-primeline"}]})
        else:
            self._json({"detail": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        CAPTURED.clear()
        CAPTURED.update(body)
        self._json({
            "choices": [{"message": {"role": "assistant",
                                     "content": "<spk:0> Guten Tag. <spk:1> Hallo zusammen."}}],
            "ctc_timestamps": {
                "error": None,
                "words": [
                    {"word": "Guten", "start": 0.1, "end": 0.4, "speaker": "<spk:0>"},
                    {"word": "Tag", "start": 0.4, "end": 0.7, "speaker": "<spk:0>"},
                    {"word": "Hallo", "start": 1.1, "end": 1.4, "speaker": "<spk:1>"},
                    {"word": "zusammen", "start": 1.4, "end": 1.9, "speaker": "<spk:1>"},
                ],
                "diarization": [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                    {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
                ],
                "speaker_tag_to_diarization_speaker": {
                    "<spk:0>": "SPEAKER_00",
                    "<spk:1>": "SPEAKER_01",
                },
            },
        })


def multipart(fields, filename, payload):
    boundary = "----shimtest"
    chunks = []
    for name, value in fields.items():
        chunks.append(
            ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
             % (boundary, name, value)).encode()
        )
    chunks.append(
        ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
         "Content-Type: audio/wav\r\n\r\n" % (boundary, filename)).encode()
    )
    chunks.append(payload)
    chunks.append(("\r\n--%s--\r\n" % boundary).encode())
    return b"".join(chunks), "multipart/form-data; boundary=%s" % boundary


def post(fields, payload):
    body, content_type = multipart(fields, "probe.wav", payload)
    request = urllib.request.Request(
        "http://127.0.0.1:%d/v1/audio/transcriptions" % SHIM_PORT,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main():
    upstream = ThreadingHTTPServer(("127.0.0.1", UPSTREAM_PORT), FakeVLLM)
    upstream.daemon_threads = True
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    env = dict(os.environ,
               SHIM_UPSTREAM="http://127.0.0.1:%d" % UPSTREAM_PORT,
               SHIM_PORT=str(SHIM_PORT),
               SHIM_MODEL="primeline/parakeet-primeline")
    shim = subprocess.Popen([sys.executable, SHIM], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % SHIM_PORT, timeout=1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise SystemExit("shim never became healthy")

        audio = wav_bytes(2.0)
        failures = []

        plain = post({"model": "primeline/parakeet-primeline", "response_format": "json"}, audio)
        for key in ("text", "segments", "language", "duration", "transcription_time",
                    "queue_wait_time", "task"):
            if key not in plain:
                failures.append("plain response missing %r" % key)
        if plain.get("text") != "Guten Tag. Hallo zusammen.":
            failures.append("plain text not stripped of speaker tags: %r" % plain.get("text"))
        if plain.get("task") != "transcribe":
            failures.append("plain task wrong: %r" % plain.get("task"))
        if abs(plain.get("duration", 0) - 2.0) > 0.01:
            failures.append("plain duration wrong: %r" % plain.get("duration"))
        segment = (plain.get("segments") or [{}])[0]
        for key in ("id", "seek", "start", "end", "text", "tokens", "temperature",
                    "avg_logprob", "compression_ratio", "no_speech_prob"):
            if key not in segment:
                failures.append("plain segment missing %r" % key)
        if CAPTURED.get("mm_processor_kwargs") != {"capture_ctc_timestamps": True}:
            failures.append("timestamps were not requested")
        if CAPTURED.get("skip_special_tokens") is not False:
            failures.append("skip_special_tokens must stay false")
        content = (CAPTURED.get("messages") or [{}])[0].get("content") or []
        if not any(part.get("type") == "audio_url" for part in content):
            failures.append("audio_url part missing from the chat request")

        # A client that still sends an old alias (or none at all) must keep
        # working: parakeet serves one model and ignores the field, so the shim
        # always sends the served name upstream.
        post({"model": "parakeet", "response_format": "json"}, audio)
        if CAPTURED.get("model") != "primeline/parakeet-primeline":
            failures.append("stale model alias was forwarded: %r" % CAPTURED.get("model"))
        post({"response_format": "json"}, audio)
        if CAPTURED.get("model") != "primeline/parakeet-primeline":
            failures.append("missing model field was not defaulted: %r" % CAPTURED.get("model"))

        diarized = post({"response_format": "diarized_json"}, audio)
        if set(diarized) != {"task", "duration", "text", "segments"}:
            failures.append("diarized keys wrong: %r" % sorted(diarized))
        speakers = [segment.get("speaker") for segment in diarized.get("segments") or []]
        if speakers != ["SPEAKER_00", "SPEAKER_01"]:
            failures.append("diarized speakers wrong: %r" % speakers)
        texts = [segment.get("text") for segment in diarized.get("segments") or []]
        if texts != ["Guten Tag.", "Hallo zusammen."]:
            failures.append("diarized segment text wrong: %r" % texts)
        if diarized.get("text") != "Guten Tag. Hallo zusammen.":
            failures.append("diarized text wrong: %r" % diarized.get("text"))

        if failures:
            print("FAIL")
            for failure in failures:
                print("  -", failure)
            return 1
        print("PASS  plain:", json.dumps(plain, ensure_ascii=False))
        print("PASS  diarized:", json.dumps(diarized, ensure_ascii=False))
        return 0
    finally:
        shim.terminate()
        upstream.shutdown()


if __name__ == "__main__":
    sys.exit(main())
