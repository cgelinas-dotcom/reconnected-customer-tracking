# Dev Environment Setup

One-time setup on your Mac. ~15 minutes.

## 1. Install Homebrew (if you don't have it)

Open Terminal, paste:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

When it finishes, it'll print 2–3 commands to run to "add brew to your PATH." Run those.

Verify:
```bash
brew --version
```

## 2. Install Python 3.11 and ffmpeg

```bash
brew install python@3.11 ffmpeg
```

Verify:
```bash
python3.11 --version    # should say 3.11.x
ffmpeg -version          # should print a version banner
```

## 3. Create the project's Python virtual environment

```bash
cd ~/Desktop/customer-tracking
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `pip install` will take a few minutes — it pulls down PyTorch and YOLO. ~2 GB.

Verify:
```bash
python -c "import cv2, ultralytics; print('OK')"
```

If that prints `OK`, you're set.

## 4. Re-activating later

Every new Terminal session, before running any project scripts:

```bash
cd ~/Desktop/customer-tracking
source .venv/bin/activate
```

You'll see `(.venv)` prepended to your prompt — that's how you know it's active.

---

## Troubleshooting

**`brew: command not found`** — you skipped the "add brew to PATH" step. Re-run the commands brew printed at the end of step 1.

**`pip install` fails on torch** — usually a network hiccup. Re-run the command.

**Anything else** — copy the error and paste it back to me.
