#!/usr/bin/env python3
"""
Generate training data for osascript-lm using Azure OpenAI REST API (httpx).

Split into focused pipelines:
  1. timers_durations   — timer for X minutes, countdown, sleep durations
  2. calendar_schedule  — events at specific times/dates, durations, days
  3. app_control        — open/close/switch/hide/force-quit apps
  4. system_control     — volume, brightness, dark mode, wifi, bluetooth, sleep
  5. files_finder       — move, copy, rename, create folders, trash, open
  6. communication      — iMessage, mail, notifications
  7. media_music        — play/pause/skip, playlists, music volume
  8. info_queries       — battery, time, wifi name, IP, disk space, hostname
  9. reminders_notes    — create reminders with dates, notes
  10. negative_oos      — out-of-scope commands → pass_to_cloud

Each pipeline generates ~1500-2500 pairs. Target: 15K-20K total.
Uses httpx directly against Azure OpenAI REST API (no SDK).
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
SEED_PATH = DATA_DIR / "seed_pairs.jsonl"

# Azure config
AZURE_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://shino-m9qsrnbv-eastus2.cognitiveservices.azure.com",
).rstrip("/")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5-mini")

CONCURRENT = 3
RATE_DELAY = 1.5  # seconds between batches


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    name: str
    output_file: str
    target_pairs: int
    system_prompt: str
    generation_prompt: str
    pairs_per_call: int = 80


PIPELINES = [
    Pipeline(
        name="timers_durations",
        output_file="data/pipe_timers.jsonl",
        target_pairs=2500,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: Timers, countdowns, and duration-based commands.

The model MUST learn exact number extraction. Every numeric value in the input must appear correctly converted in the output.

Cover these patterns with MANY numeric variations:
- "set a timer for X minutes" → do shell script "sleep (X*60)" & display notification
- "countdown for X seconds" → sleep (X)
- "remind me in X minutes" → sleep + notification
- "alert me after X hours" → sleep (X*3600)
- "timer for X and a half minutes" → sleep (X*60+30)
- "set a 90 second timer", "25 minute timer", "3 hour timer"
- "timer for one hour", "timer for half an hour", "quarter hour timer"

Numeric values to cover: 1, 2, 3, 5, 7, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 90 (minutes)
Also: 10, 15, 20, 30, 45, 60, 90, 120 (seconds)
Also: 1, 1.5, 2, 2.5, 3, 4 (hours)

Vary phrasing: casual ("set a quick 5 min timer"), formal ("please set a timer for 25 minutes"), terse ("timer 10m"), natural ("remind me in about 20 minutes"), question ("can you set a timer for 45 minutes?")

The sleep value in seconds MUST exactly match. 25 minutes = 1500 seconds. 90 seconds = 90. 2.5 hours = 9000.

Return ONLY a JSON array of {{"input": "...", "output": "..."}} objects. No markdown fences.""",
    ),
    Pipeline(
        name="calendar_schedule",
        output_file="data/pipe_calendar.jsonl",
        target_pairs=2500,
        pairs_per_call=60,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: Calendar events with PRECISE time/date extraction.

The model must correctly parse:
- Times: 8am, 9:15 AM, 2:30 PM, 12:00, 16:45, 7:30 in the morning, 3 in the afternoon
- Dates: today, tomorrow, next Monday, this Friday, next week, day after tomorrow
- Durations: 30 minute meeting, 2 hour workshop, 45 min call, 15 minute check-in, all day
- Combined: "1 hour meeting at 3pm tomorrow", "30 min standup at 9:15 AM on Monday"

Time conversion rules (MUST be exact):
- 8am → hours:8, minutes:0
- 9:15 AM → hours:9, minutes:15
- 2:30 PM → hours:14, minutes:30
- 12:00 PM → hours:12, minutes:0
- 12:30 AM → hours:0, minutes:30
- 4:45 PM → hours:16, minutes:45
- 11:00 PM → hours:23, minutes:0
- noon → hours:12, minutes:0
- midnight → hours:0, minutes:0

Duration: 30 minutes → end date:theDate + 30 * minutes, 2 hours → + 120 * minutes

Use proper AppleScript Calendar tell blocks with:
- set hours of theDate to X
- set minutes of theDate to Y
- make new event with properties {{summary:"...", start date:theDate, end date:theDate + Z * minutes}}

Cover: meetings, calls, lunches, standups, workshops, dentist, doctor, interviews, coffee chats, 1-on-1s, reviews, demos, presentations

Vary phrasing extensively. Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="app_control",
        output_file="data/pipe_apps.jsonl",
        target_pairs=1500,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: Application control — opening, closing, switching, hiding, force-quitting macOS apps.

Cover these apps: Safari, Chrome, Firefox, Finder, Terminal, iTerm, VS Code, Xcode, Slack, Discord, Spotify, Music, Mail, Messages, Notes, Reminders, Calendar, Preview, Photos, FaceTime, Zoom, Teams, Word, Excel, PowerPoint, Pages, Numbers, Keynote, TextEdit, Activity Monitor, System Preferences, App Store, Calculator, QuickTime, Figma, Sketch, Photoshop, Final Cut Pro

Actions:
- Open/launch/start: tell application "X" to activate
- Quit/close/exit: tell application "X" to quit
- Force quit: do shell script "killall X"
- Hide: tell application "System Events" to set visible of process "X" to false
- Switch to/bring to front: tell application "X" to activate
- List running: tell application "System Events" to get name of every process whose background only is false
- Quit all: various approaches

Vary phrasing: "open Safari", "launch Chrome", "start up VS Code", "fire up Slack", "close Terminal", "kill Xcode", "switch to Figma", "bring up Notes", "hide Messages", "force quit Zoom"

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="system_control",
        output_file="data/pipe_system.jsonl",
        target_pairs=2000,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: System controls with PRECISE numeric values.

1. VOLUME (0-100): "set volume to X" → set volume output volume X
   Values to cover: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
   Also: mute, unmute, volume up/down by 10
   Phrasings: "volume to 50", "set it to 75", "turn volume down to 20", "crank it up to 100", "lower volume to 30"

2. BRIGHTNESS (0-100): "brightness to X percent" → do shell script "brightness (X/100)"
   Values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

3. DARK MODE: toggle, enable, disable
   → tell app "System Events" to tell appearance preferences to set dark mode to true/false/not dark mode

4. WIFI: on/off/status → networksetup commands
5. BLUETOOTH: on/off → blueutil commands
6. SLEEP: put computer to sleep → tell app "System Events" to sleep
7. RESTART/SHUTDOWN: tell app "System Events" to restart/shut down
8. SCREENSHOT: full/area/window/clipboard → screencapture with flags
9. LOCK SCREEN: pmset displaysleepnow or CGSession lock
10. DO NOT DISTURB: toggle focus mode

Numbers in the output MUST exactly match the input. Volume 35 means 35, not 30 or 40.

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="files_finder",
        output_file="data/pipe_files.jsonl",
        target_pairs=1500,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: File system operations via Finder and shell commands.

Cover:
- Open folders: Desktop, Documents, Downloads, Applications, Home, Trash, specific paths
  → tell application "Finder" to open folder "X" of home folder
- Create folders: "make a folder called X on desktop"
  → tell application "Finder" to make new folder at desktop with properties {{name:"X"}}
- Move files: "move report.pdf to Documents"
  → tell application "Finder" to move file "report.pdf" of desktop to folder "Documents" of home folder
- Copy files: similar to move but duplicate
- Rename files: set name of file "X" to "Y"
- Trash/delete: move to trash, empty trash
- List files: do shell script "ls ~/Desktop"
- File info: size, count, modified date
- Open files: tell application "Finder" to open file "X"
- Reveal in Finder: reveal
- Compress/zip: do shell script "zip -r X.zip X"
- Create text file: do shell script "echo 'content' > ~/Desktop/file.txt"
- Show/hide hidden files: defaults write com.apple.finder AppleShowAllFiles

Vary file names, folder names, and paths. Use realistic names: report.pdf, notes.txt, photo.jpg, backup.zip, project folder, etc.

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="communication",
        output_file="data/pipe_communication.jsonl",
        target_pairs=1500,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: Messages, Mail, and Notifications.

1. iMessage/Messages:
   - "send a message to John saying hello" → tell application "Messages" to send "hello" to buddy "John"
   - "text Mom I'll be late" → send "I'll be late" to buddy "Mom"
   - "message +1234567890 hey" → send to phone number
   - "read my recent messages" → tell application "Messages" to get messages
   Vary: names, phone numbers, message content, phrasing

2. Mail:
   - "compose an email to john@example.com about the meeting" → tell application "Mail" to make new outgoing message with properties
   - "check for new mail" → tell application "Mail" to check for new mail
   - "read my unread emails" → count unread messages
   Vary: recipients, subjects, body text, CC

3. Notifications:
   - "show a notification saying X" → display notification "X" with title "Y"
   - "alert me with message X" → display alert "X"
   - "show dialog asking X" → display dialog "X"
   Vary: titles, messages, sounds

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="media_music",
        output_file="data/pipe_media.jsonl",
        target_pairs=1500,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: Music and media playback control.

1. Apple Music:
   - play/pause/stop/next/previous/shuffle
   - "play music" → tell application "Music" to play
   - "pause the music" → tell application "Music" to pause
   - "skip this song" → tell application "Music" to next track
   - "what's playing" → tell application "Music" to get {{name, artist}} of current track
   - "play the playlist Chill" → tell application "Music" to play playlist "Chill"
   - "set music volume to X" → tell application "Music" to set sound volume to X (0-100)

2. Spotify (same patterns but with Spotify):
   - tell application "Spotify" to play/pause/next track/previous track
   - get name/artist of current track

3. QuickTime/media:
   - play/pause video

Volume values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

Vary phrasing: "play some music", "hit pause", "next song", "skip track", "go back", "shuffle on", "what song is this", "turn up the music", "lower the music volume to 40"

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="info_queries",
        output_file="data/pipe_info.jsonl",
        target_pairs=1000,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: System information queries.

- Battery level: do shell script "pmset -g batt" or system_profiler
- Current time: do shell script "date"
- WiFi network name: do shell script "networksetup -getairportnetwork en0"
- IP address: do shell script "ipconfig getifaddr en0"
- Disk space: do shell script "df -h /"
- macOS version: do shell script "sw_vers -productVersion"
- Hostname: do shell script "hostname"
- Username: do shell script "whoami"
- Uptime: do shell script "uptime"
- CPU info: do shell script "sysctl -n machdep.cpu.brand_string"
- RAM: do shell script "sysctl -n hw.memsize"
- Running processes: do shell script "ps aux | head -20"
- Current app: tell application "System Events" to get name of first application process whose frontmost is true

Vary: "what's my battery at", "how much battery left", "check battery", "what time is it", "what wifi am I on", "what's my IP", "how much disk space", "what version of macOS", "what's my hostname"

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="reminders_notes",
        output_file="data/pipe_reminders_notes.jsonl",
        target_pairs=1500,
        pairs_per_call=60,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model converting natural language → AppleScript.

FOCUS: Reminders and Notes with precise time/date extraction.

1. Reminders with times:
   - "remind me to call mom at 6pm" → set hours to 18, minutes to 0
   - "reminder at 3:30 PM tomorrow" → +1 day, hours:15, minutes:30
   - "remind me at 8:45 AM to take medicine" → hours:8, minutes:45
   - "add a reminder for next Monday at 10am" → find next Monday, hours:10
   - "remind me in 2 hours to check the oven" → current date + 2 * hours

   Time extraction MUST be exact:
   - 6pm → hours:18
   - 3:30 PM → hours:15, minutes:30
   - 8:45 AM → hours:8, minutes:45
   - noon → hours:12
   - midnight → hours:0

2. Reminders without times:
   - "remind me to buy groceries" → simple reminder, no due date
   - "add to my reminders: pick up dry cleaning"

3. List/complete reminders:
   - "show my reminders" → list
   - "mark buy milk as done" → complete

4. Notes:
   - "create a note called X with content Y" → tell application "Notes"
   - "add a note titled Meeting Notes" → make new note
   - "search notes for X" → search
   - "list my notes"

Use realistic reminder content: call doctor, buy groceries, send email, pick up kids, submit report, pay bills, water plants, take medicine, check mail, schedule dentist

Return ONLY a JSON array of {{"input": "...", "output": "..."}}. No markdown.""",
    ),
    Pipeline(
        name="negative_oos",
        output_file="data/pipe_negative.jsonl",
        target_pairs=2500,
        pairs_per_call=80,
        system_prompt="You generate training data for a macOS voice assistant model. Output ONLY valid JSON arrays.",
        generation_prompt="""Generate {count} training pairs for a model that must recognize OUT-OF-SCOPE commands.

This model converts natural language → AppleScript for macOS system commands. But many user requests are NOT system commands — they require general AI reasoning, knowledge, creative writing, coding, etc.

For ALL of these out-of-scope inputs, the output should be EXACTLY:
PASS_TO_CLOUD

Categories of out-of-scope commands:
1. Knowledge questions: "what is the capital of France", "who invented the telephone", "what year did WW2 end"
2. Creative writing: "write me a poem", "draft a story about a dragon", "write a birthday message"
3. Code help: "write a Python function to sort a list", "debug this code", "explain this error"
4. Analysis: "analyze this data", "summarize this article", "compare these two things"
5. Math/reasoning: "what's 234 * 567", "solve this equation", "explain calculus"
6. Advice: "what should I have for dinner", "recommend a movie", "how do I fix a leaky faucet"
7. Conversation: "how are you", "tell me a joke", "what do you think about AI"
8. Translation: "translate hello to Spanish", "how do you say thank you in Japanese"
9. Planning: "plan a trip to Paris", "help me organize my day", "create a workout plan"
10. Research: "find research papers on climate change", "what are the latest news headlines"
11. Ambiguous that SOUND like system commands but aren't:
    - "open my mind to new possibilities" → PASS_TO_CLOUD (not an app)
    - "close the deal with the client" → PASS_TO_CLOUD (not closing an app)
    - "set the mood" → PASS_TO_CLOUD (not a system setting)
    - "check if I should buy Tesla stock" → PASS_TO_CLOUD
    - "remind me why I started this project" → PASS_TO_CLOUD (philosophical, not a Reminder)

Make the inputs natural and realistic. Include some that are tricky edge cases where the model might confuse them with real system commands.

Return ONLY a JSON array of {{"input": "...", "output": "PASS_TO_CLOUD"}}. No markdown.""",
    ),
]


# ---------------------------------------------------------------------------
# Azure REST API call via httpx
# ---------------------------------------------------------------------------

async def azure_chat(
    client: httpx.AsyncClient,
    messages: list[dict],
    temperature: float = 0.9,
    max_tokens: int = 16000,
) -> str | None:
    """Call Azure OpenAI chat completions REST API directly."""
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
    """Parse a JSON array from LLM response, handling markdown fences."""
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and "input" in d and "output" in d]
    except json.JSONDecodeError:
        pass

    # Try extracting array
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


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    pipeline: Pipeline,
):
    """Run a single data generation pipeline."""
    output_path = PROJECT_ROOT / pipeline.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing progress
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for line in f if line.strip())

    if existing >= pipeline.target_pairs:
        print(f"  [{pipeline.name}] Already have {existing}/{pipeline.target_pairs} pairs. Skipping.")
        return existing

    remaining = pipeline.target_pairs - existing
    num_calls = (remaining + pipeline.pairs_per_call - 1) // pipeline.pairs_per_call

    print(f"  [{pipeline.name}] Have {existing}, need {remaining} more. Making {num_calls} API calls.")

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

        # Tag with pipeline name and write
        with open(output_path, "a", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "input": pair["input"],
                    "output": pair["output"],
                    "pipeline": pipeline.name,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        generated += len(pairs)
        print(f"    [{pipeline.name}] Call {call_num+1}/{num_calls}: +{len(pairs)} pairs (total: {existing + generated})")

        if generated >= remaining:
            break

    final_count = existing + generated
    print(f"  [{pipeline.name}] Done. Total: {final_count} pairs.")
    return final_count


async def merge_all(pipelines: list[Pipeline]):
    """Merge all pipeline outputs into one expanded_pairs.jsonl."""
    merged_path = DATA_DIR / "expanded_pairs.jsonl"
    total = 0

    with open(merged_path, "w", encoding="utf-8") as out:
        for pipe in pipelines:
            pipe_path = PROJECT_ROOT / pipe.output_file
            if not pipe_path.exists():
                continue
            with open(pipe_path) as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total += 1

    print(f"\nMerged all pipelines → {merged_path} ({total} total pairs)")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    if not AZURE_API_KEY:
        print("Error: AZURE_OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Azure endpoint: {AZURE_ENDPOINT}")
    print(f"Model: {AZURE_MODEL}")
    print(f"API version: {AZURE_API_VERSION}")
    print(f"Pipelines: {len(PIPELINES)}")
    print(f"Target total: {sum(p.target_pairs for p in PIPELINES)} pairs\n")

    semaphore = asyncio.Semaphore(CONCURRENT)

    async with httpx.AsyncClient() as client:
        # Run pipelines sequentially (each pipeline is internally sequential
        # but respects the semaphore for rate limiting)
        for pipe in PIPELINES:
            print(f"\n{'='*60}")
            print(f"Pipeline: {pipe.name} (target: {pipe.target_pairs})")
            print(f"{'='*60}")
            await run_pipeline(client, semaphore, pipe)

    # Merge all pipeline outputs
    total = await merge_all(PIPELINES)

    # Also include seed data in the count
    seed_count = 0
    if SEED_PATH.exists():
        with open(SEED_PATH) as f:
            seed_count = sum(1 for line in f if line.strip())

    print(f"\nSeed pairs: {seed_count}")
    print(f"Expanded pairs: {total}")
    print(f"Grand total: {seed_count + total}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
