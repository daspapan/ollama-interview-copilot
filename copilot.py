#!/usr/bin/env python3
"""
Interview Copilot
Listens to the interviewer (system audio), transcribes locally with faster-whisper,
and streams a suggested answer into a small floating window.

Global hotkeys (Ctrl+Option on Mac, Ctrl+Alt elsewhere):
  H        show / hide the window
  Arrows   move the window
  T        cycle transparency
  S        screenshot -> read the question on screen and answer it
  M        "I'm speaking" on/off (records your mic, no answers while on)
  C        clear context
  P        pause / resume listening
  Q        quit

Run:  python copilot.py --profile profile.md
      python copilot.py --list-devices
"""
import argparse
import base64
import io
import json
import os
import platform
import queue
import threading
import time
from collections import deque

import numpy as np

IS_MAC = platform.system() == "Darwin"
SR = 16000
FRAME = 480                 # 30 ms frames
MIN_SPEECH_SEC = 0.4
MAX_UTTER_SEC = 30
HISTORY_LINES = 16
MOVE_STEP = 60
OPACITY_STEPS = [0.93, 0.75, 0.55, 0.35]

HOTKEY_HELP = ("⌃⌥H hide/show · ⌃⌥←↑→↓ move · ⌃⌥T transparency · ⌃⌥S screenshot · "
               "⌃⌥M I'm speaking · ⌃⌥C clear · ⌃⌥P pause · ⌃⌥Q quit")

SYSTEM_PROMPT = """You are a real-time interview assistant helping a job candidate.
You get a live transcript of the conversation. Lines marked INTERVIEWER are the interviewer,
lines marked ME are what the candidate already said out loud.

When the latest INTERVIEWER turn (read together with the lines just before it, since a question
can be split across several lines) asks the candidate something, write an answer the candidate
can say out loud right away:
- First line: a direct one or two sentence answer, first person, natural spoken English.
- Then 2 to 4 short bullet points with supporting detail or an example from the candidate's background.
- For behavioral questions, use a compact STAR shape (situation, action, result).
- For coding or system design, give the approach, the complexity, and short code or key components.
- Stay consistent with anything in the ME lines. For follow-up questions, build on what was already said.
- Ground everything in the candidate's background. Never invent employers, titles or numbers that
  are not in it. If the background has nothing relevant, give a solid general answer.
- Keep it short. The candidate is reading while talking.

If the latest turn is NOT a question or request (small talk, filler, an unfinished sentence,
the interviewer describing the company), reply with exactly: SKIP
"""

SCREENSHOT_PROMPT = """The candidate pressed the screenshot key. The image is their screen right now.
Find the question, task or problem shown on screen (coding problem, multiple choice, aptitude
question, text in a shared doc or chat, a diagram) and answer it.
- Multiple choice: give the correct option first, then one line of reasoning.
- Coding: restate the problem in one line, give the approach, time/space complexity, then clean code
  in the language shown on screen (Python if none is shown).
- Anything else: a direct answer first, then brief supporting points.
Ignore browser chrome, video tiles and other UI that is not part of the question."""


# ----------------------------------------------------------------------------- audio

def mono(data):
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def list_devices():
    import soundcard as sc
    print("Input devices (use any part of the name with --device or --my-mic):")
    for m in sc.all_microphones(include_loopback=True):
        tag = " [loopback]" if getattr(m, "isloopback", False) else ""
        print(f"  - {m.name}{tag}")
    print(f"\nDefault speaker: {sc.default_speaker().name}")
    print(f"Default microphone: {sc.default_microphone().name}")


def find_input(name_substr):
    import soundcard as sc
    for m in sc.all_microphones(include_loopback=True):
        if name_substr.lower() in m.name.lower():
            return m
    return None


def pick_source(device_substr, use_mic):
    import soundcard as sc
    if device_substr:
        m = find_input(device_substr)
        if not m:
            raise SystemExit(f"No input device matching '{device_substr}'. Try --list-devices.")
        return m
    if use_mic:
        return sc.default_microphone()
    if IS_MAC:
        m = find_input("BlackHole")
        if m:
            return m
        raise SystemExit("macOS can't record speaker output directly. Install BlackHole "
                         "(see README) and run with --device BlackHole.")
    spk = sc.default_speaker()
    try:
        return sc.get_microphone(id=str(spk.name), include_loopback=True)
    except Exception:
        raise SystemExit("Couldn't open a loopback of your speakers. Run --list-devices "
                         "and pick one with --device.")


class Listener(threading.Thread):
    """Energy-based voice activity detection on the interviewer's audio."""

    def __init__(self, source, utter_q, ui_q, paused, silence_sec, min_rms):
        super().__init__(daemon=True)
        self.source, self.utter_q, self.ui_q = source, utter_q, ui_q
        self.paused, self.silence_sec, self.min_rms = paused, silence_sec, min_rms

    def run(self):
        frame_sec = FRAME / SR
        pre_roll = deque(maxlen=int(0.3 / frame_sec))
        silence_frames = int(self.silence_sec / frame_sec)
        max_frames = int(MAX_UTTER_SEC / frame_sec)
        floor, speaking, silent, speech = None, False, 0, []

        try:
            with self.source.recorder(samplerate=SR, blocksize=FRAME) as rec:
                self.ui_q.put(("status", f"Listening to: {self.source.name}"))
                while True:
                    data = mono(rec.record(numframes=FRAME))
                    if self.paused.is_set():
                        speaking, speech, silent = False, [], 0
                        continue

                    rms = float(np.sqrt(np.mean(data * data)) + 1e-9)
                    if floor is None:
                        floor = rms
                    thr = max(self.min_rms, floor * 3.0)

                    if not speaking:
                        floor = 0.95 * floor + 0.05 * min(rms, thr)
                        pre_roll.append(data)
                        if rms > thr:
                            speaking, silent = True, 0
                            speech = list(pre_roll)
                            self.ui_q.put(("hearing", True))
                    else:
                        speech.append(data)
                        silent = silent + 1 if rms < thr * 0.7 else 0
                        if silent >= silence_frames or len(speech) >= max_frames:
                            voiced = len(speech) - silent
                            if voiced * frame_sec >= MIN_SPEECH_SEC:
                                self.utter_q.put(("them", np.concatenate(speech)))
                            speaking, speech, silent = False, [], 0
                            pre_roll.clear()
                            self.ui_q.put(("hearing", False))
        except Exception as e:
            self.ui_q.put(("status", f"Audio error: {e}"))


class MicRecorder(threading.Thread):
    """Records your own mic only while 'I'm speaking' is on, then sends it for transcription."""

    def __init__(self, source, utter_q, ui_q, talking):
        super().__init__(daemon=True)
        self.source, self.utter_q, self.ui_q, self.talking = source, utter_q, ui_q, talking

    def run(self):
        buf = []
        try:
            with self.source.recorder(samplerate=SR, blocksize=FRAME) as rec:
                while True:
                    data = mono(rec.record(numframes=FRAME * 4))
                    if self.talking.is_set():
                        buf.append(data)
                    elif buf:
                        audio = np.concatenate(buf)
                        buf = []
                        if len(audio) > SR * 0.5:
                            self.utter_q.put(("me", audio))
        except Exception as e:
            self.ui_q.put(("status", f"Mic unavailable ({e}). ⌃⌥M will still mute answers."))


# ----------------------------------------------------------------------------- speech to text

JUNK = {"", "you", "thank you.", "thanks for watching!", "bye.", ".", "uh", "um"}


class Transcriber(threading.Thread):
    def __init__(self, model_name, utter_q, on_text, ui_q, language):
        super().__init__(daemon=True)
        self.model_name, self.utter_q, self.on_text = model_name, utter_q, on_text
        self.ui_q, self.language = ui_q, language

    def run(self):
        from faster_whisper import WhisperModel
        try:
            import ctranslate2
            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda = False
        device, ctype = ("cuda", "float16") if cuda else ("cpu", "int8")
        self.ui_q.put(("status", f"Loading Whisper '{self.model_name}' on {device}..."))
        model = WhisperModel(self.model_name, device=device, compute_type=ctype)
        self.ui_q.put(("ready", None))

        while True:
            speaker, audio = self.utter_q.get()
            t0 = time.time()
            segments, _ = model.transcribe(
                audio, language=self.language, beam_size=1,
                condition_on_previous_text=False, vad_filter=(speaker == "me"),
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            if text.lower() in JUNK or len(text) < 3:
                continue
            who = "You" if speaker == "me" else "Them"
            self.ui_q.put(("heard", f"{who}: {text}  ({time.time() - t0:.1f}s)"))
            self.on_text(text, speaker)


# ----------------------------------------------------------------------------- LLM backends
# Each yields text chunks. `image` is a base64 PNG or None.

def stream_ollama(cfg, system, user, image=None, model=None):
    import requests
    msg = {"role": "user", "content": user}
    if image:
        msg["images"] = [image]
    r = requests.post(
        f"{cfg.ollama_host}/api/chat",
        json={"model": model or cfg.model, "stream": True,
              "keep_alive": "2h", "options": {"temperature": 0.3},
              "messages": [{"role": "system", "content": system}, msg]},
        stream=True, timeout=300,
    )
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        j = json.loads(line)
        if "error" in j:
            raise RuntimeError(j["error"])
        tok = j.get("message", {}).get("content", "")
        if tok:
            yield tok
        if j.get("done"):
            break


def stream_openai_compatible(cfg, system, user, image=None, model=None):
    """Works with OpenAI, Groq, LM Studio, vLLM, anything with /chat/completions."""
    import requests
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    content = user if not image else [
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
    ]
    r = requests.post(
        f"{cfg.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model or cfg.model, "stream": True, "temperature": 0.3, "max_tokens": 900,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": content}]},
        stream=True, timeout=300,
    )
    r.raise_for_status()
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", "ignore")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
        if delta:
            yield delta


def stream_anthropic(cfg, system, user, image=None, model=None):
    import anthropic
    client = anthropic.Anthropic()
    content = [{"type": "text", "text": user}]
    if image:
        content.insert(0, {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": image}})
    with client.messages.stream(
        model=model or cfg.model, max_tokens=900, system=system,
        messages=[{"role": "user", "content": content}],
    ) as s:
        for t in s.text_stream:
            yield t


BACKENDS = {"ollama": stream_ollama, "openai": stream_openai_compatible,
            "anthropic": stream_anthropic}


# ----------------------------------------------------------------------------- answer engine

class Answerer:
    """Keeps a rolling two-speaker transcript. A new question cancels the answer in flight."""

    def __init__(self, cfg, profile, ui_q, talking):
        self.cfg, self.ui_q, self.talking = cfg, ui_q, talking
        self.system = SYSTEM_PROMPT + "\n\n=== CANDIDATE BACKGROUND ===\n" + (profile or "(none given)")
        self.history = deque(maxlen=HISTORY_LINES)
        self.gen_id = 0
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.history.clear()
            self.gen_id += 1

    def on_text(self, text, speaker="them", force=False):
        label = "ME" if speaker == "me" else "INTERVIEWER"
        with self.lock:
            self.history.append(f"{label}: {text}")
            # Your own lines, or anything said while you're talking, are context only.
            if speaker == "me" or (self.talking.is_set() and not force):
                return
            self.gen_id += 1
            gid, lines = self.gen_id, list(self.history)
        transcript = "\n".join(lines[:-1]) or "(start of interview)"
        user = f"Conversation so far:\n{transcript}\n\nLatest turn:\n{lines[-1]}\n"
        if force:
            user += "\nThis was typed in manually, so treat it as a question and answer it.\n"
        self._spawn(gid, user, text, allow_skip=not force)

    def on_screenshot(self, image_b64):
        with self.lock:
            self.gen_id += 1
            gid, lines = self.gen_id, list(self.history)
        context = "\n".join(lines[-6:]) or "(none)"
        user = f"{SCREENSHOT_PROMPT}\n\nRecent conversation for context:\n{context}\n"
        self._spawn(gid, user, "[screenshot]", allow_skip=False,
                    image=image_b64, model=self.cfg.vision_model)

    def _spawn(self, *args, **kw):
        threading.Thread(target=self._run, args=args, kwargs=kw, daemon=True).start()

    def _run(self, gid, user, label, allow_skip=True, image=None, model=None):
        stream = BACKENDS[self.cfg.backend]
        started, buf = False, ""
        if image:
            self.ui_q.put(("answer_start", gid, "Reading screenshot..."))
            started = True
        try:
            for tok in stream(self.cfg, self.system, user, image=image, model=model):
                if gid != self.gen_id:
                    return
                if not started:
                    buf += tok
                    if len(buf.strip()) < 4:
                        continue
                    if allow_skip and buf.strip().upper().startswith("SKIP"):
                        return
                    started = True
                    self.ui_q.put(("answer_start", gid, label))
                    tok = buf
                self.ui_q.put(("token", gid, tok))
            if not started and buf.strip() and not (allow_skip and buf.strip().upper().startswith("SKIP")):
                self.ui_q.put(("answer_start", gid, label))
                self.ui_q.put(("token", gid, buf))
        except Exception as e:
            self.ui_q.put(("status", f"LLM error ({self.cfg.backend}): {e}"))


# ----------------------------------------------------------------------------- global hotkeys

# macOS virtual key codes
MAC_KEYS = {4: "toggle", 1: "shot", 46: "me", 8: "clear", 35: "pause", 12: "quit",
            17: "opacity", 123: "left", 124: "right", 125: "down", 126: "up"}
ARROWS = {"left", "right", "up", "down"}


def start_hotkeys(ui_q):
    if IS_MAC:
        _start_hotkeys_mac(ui_q)
    else:
        _start_hotkeys_pynput(ui_q)


def _start_hotkeys_mac(ui_q):
    """Listen-only Quartz event tap. Needs Input Monitoring permission for your terminal app."""
    import Quartz
    CTRL, ALT, CMD = (Quartz.kCGEventFlagMaskControl, Quartz.kCGEventFlagMaskAlternate,
                      Quartz.kCGEventFlagMaskCommand)
    holder = {}

    def callback(proxy, etype, event, refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(holder["tap"], True)
            return event
        if etype != Quartz.kCGEventKeyDown:
            return event
        flags = Quartz.CGEventGetFlags(event)
        if not (flags & CTRL and flags & ALT) or flags & CMD:
            return event
        code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        action = MAC_KEYS.get(code)
        if action:
            repeat = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
            if not repeat or action in ARROWS:
                ui_q.put(("hotkey", action))
        return event

    def run():
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                      Quartz.kCGEventTapOptionListenOnly, mask, callback, None)
        if tap is None:
            ui_q.put(("status", "Hotkeys OFF: allow your terminal in System Settings → Privacy & "
                                "Security → Input Monitoring, then restart the terminal."))
            return
        holder["tap"] = tap
        src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        Quartz.CFRunLoopRun()

    threading.Thread(target=run, daemon=True).start()


def _start_hotkeys_pynput(ui_q):
    try:
        from pynput import keyboard
    except ImportError:
        ui_q.put(("status", "Hotkeys OFF: pip install pynput"))
        return
    combos = {"h": "toggle", "s": "shot", "m": "me", "c": "clear", "p": "pause", "q": "quit",
              "t": "opacity", "<left>": "left", "<right>": "right", "<up>": "up", "<down>": "down"}
    mapping = {f"<ctrl>+<alt>+{k}": (lambda a=a: ui_q.put(("hotkey", a))) for k, a in combos.items()}
    keyboard.GlobalHotKeys(mapping).start()


# ----------------------------------------------------------------------------- screenshot

def grab_screen_b64(max_side=1600):
    from PIL import ImageGrab
    img = ImageGrab.grab()
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode()


# ----------------------------------------------------------------------------- UI

def run_ui(cfg, ui_q, paused, talking, answerer):
    import tkinter as tk

    BG, FG, DIM, ACC = "#111418", "#e8e8e8", "#7d8590", "#4ea1ff"
    TITLE = "Interview Copilot"
    root = tk.Tk()
    root.title(TITLE)
    root.geometry(cfg.geometry)
    root.configure(bg=BG)
    root.attributes("-topmost", True)
    state = {"visible": True, "opacity": 0, "gid": -1}

    def set_opacity(i):
        state["opacity"] = i % len(OPACITY_STEPS)
        try:
            root.attributes("-alpha", OPACITY_STEPS[state["opacity"]])
        except tk.TclError:
            pass

    set_opacity(0)

    # --- macOS: float above full-screen apps and on every Space, hide without stealing focus
    def ns_windows():
        if not IS_MAC:
            return []
        try:
            from AppKit import NSApplication
            return [w for w in NSApplication.sharedApplication().windows() if w.title() == TITLE]
        except Exception:
            return []

    def float_everywhere():
        for w in ns_windows():
            w.setLevel_(25)                      # NSStatusWindowLevel
            w.setCollectionBehavior_(1 | 256)    # all Spaces + full-screen auxiliary
    root.after(300, float_everywhere)

    def hide():
        wins = ns_windows()
        if wins:
            for w in wins:
                w.orderOut_(None)
        else:
            root.withdraw()
        state["visible"] = False

    def show():
        wins = ns_windows()
        if wins:
            for w in wins:
                w.orderFrontRegardless()
        else:
            root.deiconify()
            root.attributes("-topmost", True)
        state["visible"] = True

    def toggle_visible():
        hide() if state["visible"] else show()

    def move(dx, dy):
        root.geometry(f"+{root.winfo_x() + dx}+{root.winfo_y() + dy}")

    # --- layout
    top = tk.Frame(root, bg=BG)
    top.pack(fill="x", padx=8, pady=(8, 4))
    dot = tk.Label(top, text="●", fg=DIM, bg=BG, font=("Helvetica", 13))
    dot.pack(side="left")
    status = tk.Label(top, text="Starting...", fg=DIM, bg=BG, anchor="w", font=("Helvetica", 10))
    status.pack(side="left", fill="x", expand=True, padx=6)
    mode = tk.Label(root, text="", fg="#f85149", bg=BG, anchor="w", font=("Helvetica", 11, "bold"))
    mode.pack(fill="x", padx=8)

    def toggle_pause():
        if paused.is_set():
            paused.clear(); pause_btn.config(text="Pause"); status.config(text="Listening")
        else:
            paused.set(); pause_btn.config(text="Resume"); status.config(text="Paused")

    def toggle_talking():
        if talking.is_set():
            talking.clear(); mode.config(text="")
            dot.config(fg=DIM)
        else:
            talking.set(); mode.config(text="🎙  You're speaking (no new answers)")
            dot.config(fg="#f85149")

    def clear_all():
        answer.delete("1.0", "end"); heard.delete("1.0", "end")
        answerer.clear()

    def screenshot():
        was_visible = state["visible"]
        if was_visible:
            hide()
        root.update()

        def grab():
            try:
                img = grab_screen_b64()
            except Exception as e:
                ui_q.put(("status", f"Screenshot failed: {e}"))
                img = None
            ui_q.put(("shot_done", was_visible, img))
        root.after(250, lambda: threading.Thread(target=grab, daemon=True).start())

    def quit_app():
        try:
            root.destroy()
        finally:
            os._exit(0)

    btn = dict(bg="#22272e", fg=FG, relief="flat", activebackground="#2d333b", padx=8)
    pause_btn = tk.Button(top, text="Pause", command=toggle_pause, **btn)
    pause_btn.pack(side="right", padx=2)
    tk.Button(top, text="Clear", command=clear_all, **btn).pack(side="right", padx=2)

    heard = tk.Text(root, height=5, bg="#0b0d10", fg=DIM, wrap="word", relief="flat",
                    font=("Helvetica", 10), padx=6, pady=4)
    heard.pack(fill="x", padx=8)

    answer = tk.Text(root, bg=BG, fg=FG, wrap="word", relief="flat",
                     font=("Helvetica", cfg.font_size), padx=6, pady=6)
    answer.tag_config("q", foreground=ACC, font=("Helvetica", max(10, cfg.font_size - 2), "italic"))
    answer.pack(fill="both", expand=True, padx=8, pady=6)

    entry = tk.Entry(root, bg="#0b0d10", fg=FG, insertbackground=FG, relief="flat",
                     font=("Helvetica", 11))
    entry.pack(fill="x", padx=8, pady=(0, 4), ipady=4)
    entry.insert(0, "Type a question and press Enter")
    entry.bind("<FocusIn>", lambda e: entry.delete(0, "end"))

    tk.Label(root, text=HOTKEY_HELP, fg=DIM, bg=BG, wraplength=540, justify="left",
             font=("Helvetica", 9)).pack(fill="x", padx=8, pady=(0, 8))

    def manual(_e=None):
        q = entry.get().strip()
        if q:
            heard.insert("end", f"[typed] {q}\n"); heard.see("end")
            answerer.on_text(q, "them", force=True)
            entry.delete(0, "end")
    entry.bind("<Return>", manual)

    actions = {
        "toggle": toggle_visible, "shot": screenshot, "me": toggle_talking,
        "clear": clear_all, "pause": toggle_pause, "quit": quit_app,
        "opacity": lambda: set_opacity(state["opacity"] + 1),
        "left": lambda: move(-MOVE_STEP, 0), "right": lambda: move(MOVE_STEP, 0),
        "up": lambda: move(0, -MOVE_STEP), "down": lambda: move(0, MOVE_STEP),
    }

    def poll():
        try:
            while True:
                ev = ui_q.get_nowait()
                kind = ev[0]
                if kind == "hotkey":
                    actions[ev[1]]()
                elif kind == "status":
                    status.config(text=ev[1])
                elif kind == "ready":
                    status.config(text=f"Ready  |  {cfg.backend}: {cfg.model}")
                elif kind == "hearing" and not talking.is_set():
                    dot.config(fg="#3fb950" if ev[1] else DIM)
                elif kind == "heard":
                    heard.insert("end", ev[1] + "\n"); heard.see("end")
                elif kind == "shot_done":
                    if ev[1]:
                        show()
                    if ev[2]:
                        answerer.on_screenshot(ev[2])
                elif kind == "answer_start":
                    state["gid"] = ev[1]
                    answer.delete("1.0", "end")
                    answer.insert("end", f"Q: {ev[2]}\n\n", "q")
                elif kind == "token" and ev[1] == state["gid"]:
                    answer.insert("end", ev[2]); answer.see("end")
        except queue.Empty:
            pass
        root.after(30, poll)

    root.protocol("WM_DELETE_WINDOW", quit_app)
    poll()
    root.mainloop()


# ----------------------------------------------------------------------------- main

DEFAULT_MODELS = {
    "ollama": ("llama3.2:3b", "gemma3:4b"),
    "anthropic": ("claude-haiku-5-5", "claude-haiku-5-5"),
    "openai_groq": ("llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "openai": ("gpt-4o-mini", "gpt-4o-mini"),
}


def main():
    p = argparse.ArgumentParser(description="Live interview answer assistant")
    p.add_argument("--profile", help="Markdown/text file with your resume + job description")
    p.add_argument("--backend", choices=BACKENDS, default=os.environ.get("COPILOT_BACKEND", "ollama"))
    p.add_argument("--model", default=os.environ.get("COPILOT_MODEL"), help="Model for spoken questions")
    p.add_argument("--vision-model", default=os.environ.get("COPILOT_VISION_MODEL"),
                   help="Model for screenshots (must support images)")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"))
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--whisper", default="base.en",
                   help="tiny.en / base.en / small.en / medium.en / large-v3")
    p.add_argument("--language", default="en")
    p.add_argument("--device", help="Interviewer audio input, by name (Mac: BlackHole)")
    p.add_argument("--mic", action="store_true", help="Use the default mic as the interviewer source")
    p.add_argument("--my-mic", help="Your microphone for ⌃⌥M, by name (default: system mic)")
    p.add_argument("--silence", type=float, default=0.7, help="Pause length (s) that ends a question")
    p.add_argument("--min-rms", type=float, default=0.008, help="Raise if background noise triggers it")
    p.add_argument("--font-size", type=int, default=14)
    p.add_argument("--geometry", default="560x660+40+40")
    p.add_argument("--list-devices", action="store_true")
    cfg = p.parse_args()

    if cfg.list_devices:
        list_devices(); return

    key = cfg.backend
    if key == "openai":
        key = "openai_groq" if "groq" in cfg.base_url else "openai"
    text_default, vision_default = DEFAULT_MODELS[key]
    cfg.model = cfg.model or text_default
    cfg.vision_model = cfg.vision_model or vision_default

    profile = ""
    if cfg.profile:
        with open(cfg.profile, encoding="utf-8") as f:
            profile = f.read()

    ui_q, utter_q = queue.Queue(), queue.Queue()
    paused, talking = threading.Event(), threading.Event()
    answerer = Answerer(cfg, profile, ui_q, talking)

    import soundcard as sc
    source = pick_source(cfg.device, cfg.mic)
    my_mic = find_input(cfg.my_mic) if cfg.my_mic else sc.default_microphone()

    Transcriber(cfg.whisper, utter_q, answerer.on_text, ui_q, cfg.language).start()
    Listener(source, utter_q, ui_q, paused, cfg.silence, cfg.min_rms).start()
    if my_mic is not None and my_mic.name != source.name:
        MicRecorder(my_mic, utter_q, ui_q, talking).start()
    start_hotkeys(ui_q)
    run_ui(cfg, ui_q, paused, talking, answerer)


if __name__ == "__main__":
    main()
