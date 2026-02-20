#!/usr/bin/env python3
"""
Generate CONVERSATIONAL / VERBOSE training data for osascript-lm.

The model handles short commands well but struggles with longer, more natural
phrasing like "hey can you open Chrome for me" or "I want to put my computer
to sleep now". This script generates longer variants across all categories.

Uses httpx directly against Azure OpenAI REST API.
"""

import asyncio
import json
import os
import sys
import time as time_mod
from pathlib import Path
from dataclasses import dataclass

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Azure config
AZURE_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://shino-m9qsrnbv-eastus2.cognitiveservices.azure.com",
).rstrip("/")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5-mini")

CONCURRENT = 3
RATE_DELAY = 1.5


@dataclass
class Pipeline:
    name: str
    output_file: str
    target_pairs: int
    system_prompt: str
    generation_prompt: str
    pairs_per_call: int = 60


SYSTEM_PROMPT = "You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays."

PIPELINES = [
    Pipeline(
        name="conv_timers",
        output_file="data/pipe_conv_timers.jsonl",
        target_pairs=1500,
        pairs_per_call=60,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: LONGER, CONVERSATIONAL timer and countdown commands.

The model already handles short commands like "set a 5 minute timer" perfectly.
Now we need VERBOSE, NATURAL-SOUNDING versions that real people would say in conversation.

Each input MUST be 8+ words. Use filler words, politeness, context, and varied sentence structures:
- "hey can you set a timer for 20 minutes please"
- "I need a countdown for about 45 minutes if you don't mind"
- "could you start a 10 minute timer for me real quick"
- "please set up a timer I need 30 minutes for my workout"
- "go ahead and start a 15 minute countdown timer"
- "I'm cooking something can you set a timer for 25 minutes"
- "would you mind setting a 3 hour timer for me"
- "alright set me a timer for one hour and fifteen minutes"
- "start a quick 5 minute timer while I wait"
- "I want a countdown timer set to 90 seconds please"

The output AppleScript MUST use do shell script "sleep N" (NOT delay). Correct time math:
- X minutes → do shell script "sleep (X*60)" then newline display notification
- X hours → do shell script "sleep (X*3600)" then newline display notification
- X seconds → do shell script "sleep X" then newline display notification
- one hour and thirty minutes → do shell script "sleep 5400"

Example output format (MUST follow exactly):
do shell script "sleep 1200"
display notification "Timer finished" with title "Timer"

NEVER use "delay". Always use do shell script "sleep N".

Vary the conversational style: polite requests, casual commands, sentences with context ("I'm cooking..."), questions ("can you..."), statements ("I need...").

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="conv_apps",
        output_file="data/pipe_conv_apps.jsonl",
        target_pairs=1500,
        pairs_per_call=60,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: LONGER, CONVERSATIONAL app control commands.

The model handles "open Safari" perfectly. Now we need VERBOSE, NATURAL versions:

Each input MUST be 8+ words. Examples of the style we want:
- "hey open up Google Chrome for me please" → tell application "Google Chrome" to activate
- "I need you to launch the Terminal app" → tell application "Terminal" to activate
- "can you close Safari for me I'm done with it" → tell application "Safari" to quit
- "go ahead and open Slack I need to check my messages" → tell application "Slack" to activate
- "would you mind opening up VS Code for me" → tell application "Visual Studio Code" to activate
- "please quit all the apps I have open right now" → quit all apps script
- "I want to switch over to Finder real quick" → tell application "Finder" to activate
- "could you force quit this app it's not responding" → force quit frontmost
- "open up the Notes app I need to write something down" → tell application "Notes" to activate
- "can you hide all the other windows except this one" → hide others

Apps to include: Safari, Chrome, Firefox, Terminal, VS Code, Slack, Discord, Spotify, Music, Finder, Notes, Calculator, Preview, Pages, Numbers, Keynote, Xcode, TextEdit, Mail, Messages, FaceTime, Calendar, Reminders, Photos, Maps, System Preferences, Activity Monitor

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="conv_system",
        output_file="data/pipe_conv_system.jsonl",
        target_pairs=2000,
        pairs_per_call=60,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: LONGER, CONVERSATIONAL system control commands.

The model handles "set volume to 50" perfectly. Now we need VERBOSE, NATURAL versions:

Each input MUST be 8+ words. Cover ALL these categories with conversational phrasing:

VOLUME:
- "hey can you turn the volume up to like 70 percent" → set volume output volume 70
- "please lower the volume a bit to about 30" → set volume output volume 30
- "I need you to mute my computer real quick" → set volume with output muted
- "could you unmute the sound on my mac please" → set volume without output muted
- "set the volume to around 50 percent for me" → set volume output volume 50

BRIGHTNESS:
- "can you make the screen a little brighter please" → do shell script "brightness 0.8"
- "turn down the brightness to about 40 percent" → do shell script "brightness 0.4"
- "the screen is too bright can you dim it a bit" → do shell script "brightness 0.3"
- "set my screen brightness to 75 percent please" → do shell script "brightness 0.75"

DARK MODE:
- "hey turn on dark mode on my mac please" → tell application "System Events" to tell appearance preferences to set dark mode to true
- "switch to light mode I can't see well in dark" → set dark mode to false
- "can you enable dark mode it's getting late" → set dark mode to true

WIFI/BLUETOOTH:
- "go ahead and turn off the wifi on my mac" → do shell script "networksetup -setairportpower en0 off"
- "please turn on bluetooth I need to connect my airpods" → do shell script "blueutil --power 1"
- "can you disable wifi for me real quick" → networksetup off
- "turn bluetooth off please I don't need it" → blueutil --power 0

SLEEP/LOCK/SCREENSHOT:
- "I'm stepping away can you lock my screen" → keystroke "q" using {{command down, control down}}
- "put my mac to sleep please I'm heading out" → tell application "System Events" to sleep
- "can you take a screenshot and save it to my desktop" → screencapture
- "I want to lock my computer before I leave" → keystroke "q" using {{command down, control down}}

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="conv_music",
        output_file="data/pipe_conv_music.jsonl",
        target_pairs=1000,
        pairs_per_call=60,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: LONGER, CONVERSATIONAL music and media commands.

Each input MUST be 8+ words. Examples:
- "hey can you play some music for me please" → tell application "Music" to play
- "go ahead and pause the music real quick" → tell application "Music" to pause
- "skip to the next song in my playlist please" → tell application "Music" to next track
- "can you go back to the previous track I liked that one" → tell application "Music" to previous track
- "please stop the music I need to focus now" → tell application "Music" to stop
- "turn on shuffle mode for my music please" → tell application "Music" to set shuffle enabled to true
- "could you resume playing the music from where it stopped" → tell application "Music" to play
- "I want to set the music volume to about 60 percent" → tell application "Music" to set sound volume to 60
- "lower the music volume a bit it's too loud" → tell application "Music" to set sound volume to 30
- "play my favorites playlist in the Music app" → tell application "Music" to play playlist "Favorites"

Include: play, pause, stop, next track, previous track, shuffle on/off, repeat on/off, set music volume, play specific playlist. Use conversational fillers.

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="conv_info",
        output_file="data/pipe_conv_info.jsonl",
        target_pairs=1000,
        pairs_per_call=60,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: LONGER, CONVERSATIONAL system information queries.

Each input MUST be 8+ words. Examples:
- "hey can you tell me how much battery I have left" → do shell script "pmset -g batt"
- "I want to know what my IP address is right now" → do shell script "ipconfig getifaddr en0"
- "could you check what time it is on my mac" → do shell script "date"
- "what wifi network am I currently connected to" → do shell script "networksetup -getairportnetwork en0"
- "can you tell me how much disk space I have left" → do shell script "df -h /"
- "I need to know what version of macOS I'm running" → do shell script "sw_vers -productVersion"
- "hey what's the name of this computer" → do shell script "hostname"
- "can you check how long my mac has been running" → do shell script "uptime"
- "I want to see what my CPU is" → do shell script "sysctl -n machdep.cpu.brand_string"
- "tell me how much RAM this machine has please" → do shell script "sysctl -n hw.memsize"
- "check what user account I'm logged into" → do shell script "whoami"
- "what processes are currently running on my mac" → do shell script "ps aux | head -20"
- "which app is in the foreground right now" → tell application "System Events" to get name of first application process whose frontmost is true

Vary: polite, casual, question form, statement form. Always 8+ words.

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="conv_files",
        output_file="data/pipe_conv_files.jsonl",
        target_pairs=1000,
        pairs_per_call=60,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: LONGER, CONVERSATIONAL file and Finder commands.

Each input MUST be 8+ words. Examples:
- "can you open my Documents folder in Finder for me" → tell application "Finder" to open folder "Documents" of home
- "create a new folder called Projects on my desktop please" → tell application "Finder" to make new folder at desktop with properties {{name:"Projects"}}
- "move the file report.pdf from my desktop to Documents" → tell application "Finder" to move file "report.pdf" of desktop to folder "Documents" of home
- "how many files do I have on my desktop right now" → tell application "Finder" to return count of files of desktop
- "go ahead and empty the trash for me please" → tell application "Finder" to empty the trash
- "can you show me what's in my Downloads folder" → tell application "Finder" to open folder "Downloads" of home
- "please move that file to the trash I don't need it" → tell application "Finder" to move file to trash
- "I need you to open the Applications folder for me" → tell application "Finder" to open folder "Applications" of startup disk
- "rename the file on my desktop to something else" → tell application "Finder" to set name of file
- "show me the contents of my home directory please" → tell application "Finder" to open home

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="conv_cloud",
        output_file="data/pipe_conv_cloud.jsonl",
        target_pairs=2000,
        pairs_per_call=60,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt="""Generate {count} training pairs for a model that must recognize OUT-OF-SCOPE commands.

For ALL of these inputs, the output MUST be exactly: PASS_TO_CLOUD

FOCUS: LONGER, CONVERSATIONAL versions of cloud-routed commands (8+ words each).

These are things that SOUND like they could be system commands because of their conversational nature, but they require cloud AI:

MESSAGING (always PASS_TO_CLOUD):
- "hey can you send a text to John saying I'll be late"
- "I need you to message Sarah and tell her about the meeting"
- "could you text my mom and let her know I'm on my way home"

EMAIL (always PASS_TO_CLOUD):
- "please send an email to my boss about the deadline"
- "I need to write an email to the team about tomorrow's standup"
- "can you draft a quick email to Sarah about the project update"

CALENDAR (always PASS_TO_CLOUD):
- "add a meeting to my calendar for tomorrow at 3pm please"
- "I need you to schedule a dentist appointment for next Monday"
- "can you create a calendar event for the team lunch on Friday"

REMINDERS (always PASS_TO_CLOUD):
- "remind me to pick up the kids from school at 4pm"
- "I need a reminder to call the dentist tomorrow morning"
- "can you add a reminder to buy groceries on my way home"

TASKS (always PASS_TO_CLOUD):
- "create a to-do list for the things I need to do this week"
- "add finish the report to my task list please"
- "can you help me plan out my tasks for today"

KNOWLEDGE/CREATIVE (always PASS_TO_CLOUD):
- "hey can you explain how machine learning works to me"
- "I want you to write a birthday message for my friend"
- "could you help me figure out what to cook for dinner tonight"
- "tell me about the history of the Roman Empire please"
- "can you summarize this article I just read about climate change"

Make inputs sound natural and conversational. Many should use filler words and polite phrasing that could trick the model into thinking they're system commands.

Return ONLY a JSON array of {{"input": "...", "output": "PASS_TO_CLOUD"}}. No markdown.""",
    ),
]


# ---------------------------------------------------------------------------
# Azure REST API call via httpx
# ---------------------------------------------------------------------------

async def azure_chat(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 16000,
) -> str | None:
    url = (
        f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_MODEL}"
        f"/chat/completions?api-version={AZURE_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY,
    }
    payload = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }

    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "30"))
            print(f"    Rate limited. Waiting {retry_after}s...")
            await asyncio.sleep(retry_after)
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)

        if resp.status_code != 200:
            print(f"    API error {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except httpx.TimeoutException:
        print("    Request timed out")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def parse_json_response(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and "input" in d and "output" in d]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and "input" in d and "output" in d]
        except json.JSONDecodeError:
            pass

    print("    Failed to parse JSON response")
    return []


async def run_pipeline(client, semaphore, pipeline):
    output_path = PROJECT_ROOT / pipeline.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for line in f if line.strip())

    if existing >= pipeline.target_pairs:
        print(f"  [{pipeline.name}] Already have {existing}/{pipeline.target_pairs}. Skipping.")
        return existing

    remaining = pipeline.target_pairs - existing
    num_calls = (remaining + pipeline.pairs_per_call - 1) // pipeline.pairs_per_call

    print(f"  [{pipeline.name}] Have {existing}, need {remaining} more. {num_calls} API calls.")

    generated = 0
    for call_num in range(num_calls):
        count = min(pipeline.pairs_per_call, remaining - generated)
        prompt = pipeline.generation_prompt.format(count=count)

        async with semaphore:
            text = await azure_chat(
                client,
                messages=[
                    {"role": "system", "content": pipeline.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            await asyncio.sleep(RATE_DELAY)

        if text is None:
            continue

        pairs = parse_json_response(text)
        if not pairs:
            continue

        with open(output_path, "a", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "input": pair["input"],
                    "output": pair["output"],
                    "pipeline": pipeline.name,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        generated += len(pairs)
        print(f"    [{pipeline.name}] Call {call_num+1}/{num_calls}: +{len(pairs)} (total: {existing + generated})")

        if generated >= remaining:
            break

    final_count = existing + generated
    print(f"  [{pipeline.name}] Done. Total: {final_count}")
    return final_count


async def main():
    if not AZURE_API_KEY:
        print("Error: AZURE_OPENAI_API_KEY not set.", file=sys.stderr)
        print("Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)

    print(f"Azure endpoint: {AZURE_ENDPOINT}")
    print(f"Model: {AZURE_MODEL}")
    print(f"Pipelines: {len(PIPELINES)}")
    print(f"Target total: {sum(p.target_pairs for p in PIPELINES)} pairs\n")

    semaphore = asyncio.Semaphore(CONCURRENT)

    async with httpx.AsyncClient() as client:
        for pipe in PIPELINES:
            print(f"\n{'='*60}")
            print(f"Pipeline: {pipe.name} (target: {pipe.target_pairs})")
            print(f"{'='*60}")
            await run_pipeline(client, semaphore, pipe)

    # Merge conversational data into a single file
    merged_path = DATA_DIR / "expanded_conversational.jsonl"
    total = 0
    with open(merged_path, "w", encoding="utf-8") as out:
        for pipe in PIPELINES:
            pipe_path = PROJECT_ROOT / pipe.output_file
            if not pipe_path.exists():
                continue
            with open(pipe_path) as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total += 1

    print(f"\nMerged → {merged_path} ({total} conversational pairs)")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
