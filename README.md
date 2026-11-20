# Interview Copilot

Listens to the interviewer's audio, transcribes it locally with Whisper, and streams a suggested answer into a floating window that stays above every app (including full-screen Zoom/Meet) and on every desktop Space.

## Hotkeys

All hotkeys are **Control + Option** plus a key on Mac (Ctrl + Alt on Windows/Linux). They work no matter which app is focused.

| Keys | What it does |
|---|---|
| ⌃⌥ H | Show / hide the window (doesn't steal focus from your call) |
| ⌃⌥ ← ↑ → ↓ | Move the window (hold to keep moving). You can also drag it by the title bar |
| ⌃⌥ T | Cycle transparency: 93% → 75% → 55% → 35% |
| ⌃⌥ S | Screenshot: hides the window, captures the screen, reads the question on it and answers (coding problems, MCQs, shared docs) |
| ⌃⌥ M | "I'm speaking" on/off. Records your mic while on, adds what you said to the context so follow-up answers stay consistent, and stops new answers from popping up while you talk |
| ⌃⌥ C | Clear the conversation context (new interview) |
| ⌃⌥ P | Pause / resume listening |
| ⌃⌥ Q | Quit |

## Mac (M1/M2/M3) setup with Ollama

**1. Install**
```bash
brew install python@3.12 python-tk@3.12 ollama
brew install --cask blackhole-2ch          # restart the Mac afterwards
```

**2. Models.** One model for both spoken questions and screenshots keeps memory low:
```bash
ollama serve &              # or open the Ollama app
ollama pull gemma3:4b       # text + vision, fits an 8 GB Mac
```
With 16 GB+ you can use `llama3.1:8b` for speech (`--model llama3.1:8b`) and keep `gemma3:4b` for screenshots.

**3. Route call audio.** Open *Audio MIDI Setup* → **+** → *Create Multi-Output Device* → tick your headphones first, then *BlackHole 2ch* (enable Drift Correction on BlackHole). Select that Multi-Output Device as your output in System Settings → Sound, or as the speaker in Zoom/Meet. Set your headphone volume before switching, since the volume keys don't work on Multi-Output devices.

**4. Install and run**
```bash
cd interview-copilot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python copilot.py --profile profile.md --model gemma3:4b
```

**5. Permissions.** In System Settings → Privacy & Security, allow your terminal app (Terminal, iTerm, VS Code) under:

- **Microphone**: to read BlackHole and your mic
- **Input Monitoring**: for the global hotkeys
- **Screen & System Audio Recording**: for ⌃⌥S (without it you'll only capture the wallpaper)

Quit and reopen the terminal after granting these. If hotkeys don't work, the status line in the window tells you which permission is missing.

## Other LLM options

```bash
python copilot.py --backend openai                         # Groq (set GROQ_API_KEY), fastest
python copilot.py --backend openai --base-url https://api.openai.com/v1   # OpenAI (OPENAI_API_KEY)
python copilot.py --backend anthropic                      # Claude API (ANTHROPIC_API_KEY, separate from the Pro plan)
```

## Tuning

| Problem | Fix |
|---|---|
| Cuts questions off mid-sentence | `--silence 1.2` |
| Background noise triggers it | `--min-rms 0.02` |
| Transcripts are inaccurate | `--whisper small.en` |
| Wrong mic used for ⌃⌥M | `--my-mic "MacBook"` (see `--list-devices`) |
| Window size/position at start | `--geometry 500x700+1300+50 --font-size 15` |
| First answer slow | Type a test question in the box before the call to load the model |



Verson 2
Core fixes for answers changing every sentence

Tell the four kinds apart: new question, follow-up, more detail, or not a question. The model labels each interviewer turn with the first word of its reply, so this adds no extra delay.
New question: the answer on screen is replaced.
Follow-up: the previous answer shrinks to its one-line core answer and the follow-up answer appears below it, labelled “↳ Follow-up”.
More detail on the same question: one or two “Also:” bullets are added at the bottom, and the existing points are never rewritten.
Not a question: nothing changes.
Keep the answer stable once it’s shown. An answer that has started showing is never wiped halfway. It’s only replaced when a genuinely new question arrives.
Wait a little longer before answering. The copilot waits 1.2 seconds of silence instead of 0.7, so it stops answering half a question. The first word arrives about half a second later.
Make the wording consistent. Lower the model’s randomness (temperature 0.3 to 0). If an answer does get regenerated, it says nearly the same thing in nearly the same words.
Add ⌃⌥R: refresh the answer. This forces a fresh answer to the latest question, for when the app wrongly keeps an old one.
Show the kind of turn in the status line. For example “New question”, “Follow-up”, “Added detail” or “Ignored”, so you can tell at a glance what the app decided.