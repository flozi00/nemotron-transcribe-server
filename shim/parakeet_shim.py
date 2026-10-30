#!/usr/bin/env python3
"""Parakeet-compatible front end for a vLLM Nemotron-Transcribe server.

The pool's serving port is answered here; everything is forwarded to vLLM on
127.0.0.1:<SHIM_UPSTREAM port>. ``POST /v1/audio/transcriptions`` is the one
translated route: parakeet-server's multipart request becomes a
``/v1/chat/completions`` call with the audio inline as a data URL, and the
answer is reshaped into parakeet-server's response schema so an existing
client keeps working unchanged.

Stdlib only, like the console's own proxy: no pip dependency in the image.
"""

import base64
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from email.parser import BytesParser
from email import policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("SHIM_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
PORT = int(os.environ.get("SHIM_PORT", "8080"))
MODEL = os.environ.get("SHIM_MODEL", "primeline/parakeet-primeline")
# A dense one-hour conversation needs about 20k output tokens (model card).
MAX_TOKENS = int(os.environ.get("SHIM_MAX_TOKENS", "32768"))
# The `language` field parakeet reports. Parakeet detects it; this model can
# identify a language but only when asked as its own task, and a prompt that
# asks for the code inline is ignored (measured), so a second call would be
# needed for a field no client in this deployment branches on. The client's
# own `language` wins when it sends one; otherwise this is the deployment's
# language, which is what the pool is for.
LANGUAGE = os.environ.get("SHIM_LANGUAGE", "de")

TRANSCRIBE_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including "
    "fillers and repetitions. Formatting: punctuation and capitalization. "
    "Output: transcript only."
)
DIARIZE_PROMPT = (
    "Transcribe this audio with punctuation and capitalization. "
    "Include all spoken words and verbal fillers. "
    "Write numbers in their standard written form. "
    "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)

_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".webm": "audio/webm",
}


def _mime_for(filename):
    _, _, ext = (filename or "").rpartition(".")
    return _MIME.get("." + ext.lower(), "audio/wav")


def wav_duration(payload):
    """Seconds of audio in a RIFF/WAVE payload, or 0.0 when it is not one."""
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return 0.0
    pos, byte_rate = 12, 0
    while pos + 8 <= len(payload):
        chunk_id = payload[pos:pos + 4]
        (size,) = struct.unpack("<I", payload[pos + 4:pos + 8])
        if chunk_id == b"fmt " and size >= 16:
            (_fmt, _ch, _rate, byte_rate) = struct.unpack("<HHII", payload[pos + 8:pos + 20])
        elif chunk_id == b"data":
            if byte_rate:
                return round(min(size, len(payload) - pos - 8) / byte_rate, 3)
            return 0.0
        pos += 8 + size + (size & 1)
    return 0.0


def parse_multipart(body, content_type):
    """(fields, files) from a multipart/form-data body; files are name -> list."""
    head = b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n"
    message = BytesParser(policy=policy.default).parsebytes(head + body)
    fields, files = {}, {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is None:
            fields.setdefault(name, []).append(payload.decode("utf-8", "replace"))
        else:
            files.setdefault(name, []).append((filename, payload))
    return fields, files


def _first(fields, name, default=None):
    values = fields.get(name)
    return values[0] if values else default


def _post_json(path, payload, timeout):
    request = urllib.request.Request(
        UPSTREAM + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def _strip_speaker_tags(text):
    return re.sub(r"\s+", " ", re.sub(r"<spk:\d+>", " ", text or "")).strip()


def _speaker_runs(text):
    """[(tag, spoken text)] from a transcript carrying <spk:N> markers."""
    runs, current, buffer = [], None, []
    for piece in re.split(r"(<spk:\d+>)", text or ""):
        if re.fullmatch(r"<spk:\d+>", piece or ""):
            if current is not None and "".join(buffer).strip():
                runs.append((current, " ".join("".join(buffer).split())))
            current, buffer = piece, []
        else:
            buffer.append(piece)
    if current is not None and "".join(buffer).strip():
        runs.append((current, " ".join("".join(buffer).split())))
    return runs


def _speaker_key(value):
    """`<spk:0>`, `0` and `"0"` are the same speaker.

    The transcript carries `<spk:N>` tags while the CTC alignment reports the
    bare index (`"speaker": "0"`), and the tag→diarization map is keyed by the
    bare index too. Comparing the raw values matches nothing on live output,
    which silently degrades every diarized segment to the whole file.
    """
    text = str(value if value is not None else "")
    match = re.fullmatch(r"<spk:(\d+)>", text)
    return match.group(1) if match else text


def _segments_from_alignment(alignment, raw_text, duration):
    """parakeet's diarized segments: transcript text per speaker, timed by the CTC alignment.

    Speaker labels are single letters in order of first appearance (`A`, `B`,
    …), which is exactly what parakeet-server's own diarization emits — a
    client that groups by `speaker` must not see a different vocabulary.
    """
    runs = _speaker_runs(raw_text)
    if not runs:
        return None
    words = (alignment or {}).get("words") or []
    labels = {}
    segments = []
    for tag, spoken in runs:
        key = _speaker_key(tag)
        timed = [word for word in words if _speaker_key(word.get("speaker")) == key]
        if timed:
            start = min(float(word.get("start") or 0.0) for word in timed)
            end = max(float(word.get("end") or 0.0) for word in timed)
        else:
            start, end = 0.0, duration
        label = labels.setdefault(key, chr(ord("A") + len(labels)))
        segments.append({
            "id": str(len(segments)),
            "type": "transcript.text.segment",
            "speaker": label,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": spoken,
        })
    return segments or None


def _plain_segments(text, duration, alignment=None):
    """parakeet's non-diarized segments: one per sentence, timed by the CTC alignment.

    Prod parakeet splits a transcript at sentence boundaries and gives each
    segment its own start/end, so a client that renders subtitles or seeks by
    segment keeps working. The CTC head gives word-level timings; grouping
    them at sentence-ending punctuation reproduces that shape. Without an
    alignment (or when it does not cover the text) this degrades to the single
    whole-file segment it used to return.
    """
    words = (alignment or {}).get("words") or []
    grouped, current = [], []
    for word in words:
        piece = str(word.get("word") or "").strip()
        if not piece:
            continue
        current.append(word)
        if piece[-1] in ".!?":
            grouped.append(current)
            current = []
    if current:
        grouped.append(current)

    segments = []
    for group in grouped:
        spoken = " ".join(str(word.get("word") or "").strip() for word in group).strip()
        if not spoken:
            continue
        segments.append({
            "id": len(segments),
            "seek": 0,
            "start": round(float(group[0].get("start") or 0.0), 3),
            "end": round(float(group[-1].get("end") or 0.0), 3),
            "text": spoken,
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
        })
    if segments:
        return segments
    return [{
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": round(duration, 3),
        "text": (text or "").strip(),
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": None,
        "compression_ratio": None,
        "no_speech_prob": None,
    }]


def transcribe(fields, files):
    """One parakeet transcription request -> parakeet's response body."""
    uploads = files.get("file") or []
    if not uploads:
        raise ValueError("file is required")
    filename, payload = uploads[0]
    if not payload:
        raise ValueError("audio file is empty")

    response_format = _first(fields, "response_format") or "json"
    diarize = response_format == "diarized_json"
    language = _first(fields, "language") or LANGUAGE

    audio_url = "data:%s;base64,%s" % (_mime_for(filename), base64.b64encode(payload).decode())
    body = {
        # Always the served name, never the client's. Parakeet's server serves
        # exactly one model and ignores the field, so a client that still sends
        # an old alias (or none) must keep working; forwarding it verbatim would
        # make vLLM answer 404 "model not found".
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": DIARIZE_PROMPT if diarize else TRANSCRIBE_PROMPT},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ],
        }],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "seed": 0,
        "mm_processor_kwargs": {"capture_ctc_timestamps": True},
    }

    started = time.time()
    answer = _post_json("/v1/chat/completions", body, timeout=1800)
    elapsed = round(time.time() - started, 3)

    choices = answer.get("choices") or [{}]
    raw_text = ((choices[0].get("message") or {}).get("content")) or ""
    text = _strip_speaker_tags(raw_text)
    alignment = answer.get("ctc_timestamps")
    duration = wav_duration(payload)
    if not duration and isinstance(alignment, dict):
        words = alignment.get("words") or []
        if words:
            duration = round(max(float(word.get("end") or 0.0) for word in words), 3)

    if diarize:
        segments = _segments_from_alignment(alignment, raw_text, duration) or []
        if not segments and text:
            segments = [{
                "id": "0",
                "type": "transcript.text.segment",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": round(duration, 3),
                "text": text,
            }]
        return {
            "task": "transcribe",
            "duration": duration,
            "text": " ".join(segment["text"] for segment in segments) or text,
            "segments": segments,
        }

    return {
        "text": text,
        "segments": _plain_segments(text, duration, alignment),
        "language": language,
        "duration": duration,
        "transcription_time": elapsed,
        "queue_wait_time": 0.0,
        "task": "transcribe",
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "parakeet-shim/1.0"

    def log_message(self, fmt, *args):  # keep the pod log readable
        sys.stderr.write("shim: " + (fmt % args) + "\n")

    def _send(self, status, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _upstream_ok(self):
        try:
            with urllib.request.urlopen(UPSTREAM + "/health", timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in ("host", "content-length", "connection")
        }
        request = urllib.request.Request(
            UPSTREAM + self.path, data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in ("content-length", "transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (urllib.error.URLError, OSError) as error:
            self._send(503, {"detail": "model server not reachable: %s" % error})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            ready = self._upstream_ok()
            self._send(200 if ready else 503, {
                "status": "ready" if ready else "not_ready",
                "model_loaded": ready,
                "model_name": MODEL,
            })
            return
        if path == "/ready":
            ready = self._upstream_ok()
            self._send(200 if ready else 503, {"status": "ready" if ready else "not_ready"})
            return
        self._proxy("GET")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/audio/transcriptions", "/audio/transcriptions"):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            content_type = self.headers.get("Content-Type") or ""
            if "multipart/form-data" not in content_type:
                self._send(400, {"detail": "expected multipart/form-data"})
                return
            try:
                fields, files = parse_multipart(body, content_type)
                self._send(200, transcribe(fields, files))
            except ValueError as error:
                self._send(400, {"detail": str(error)})
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                self._send(error.code, {"detail": detail})
            except Exception as error:  # noqa: BLE001 - the client gets a 500, not a dead pod
                self.log_message("transcription failed: %s", error)
                self._send(500, {"detail": str(error)})
            return
        self._proxy("POST")


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    sys.stderr.write("shim: listening on 0.0.0.0:%d -> %s\n" % (PORT, UPSTREAM))
    server.serve_forever()


if __name__ == "__main__":
    main()
