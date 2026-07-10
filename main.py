from fastapi import FastAPI, Request
import anthropic
import requests
import asyncio
import os

app = FastAPI()
client = anthropic.Anthropic()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
SPOTIO_CLIENT_ID = os.environ.get("SPOTIO_CLIENT_ID")
SPOTIO_CLIENT_SECRET = os.environ.get("SPOTIO_CLIENT_SECRET")

SPOTIO_BASE = "https://api.spotio2.com"

ALLOWED_TEST_IDS = [x.strip() for x in os.environ.get("ALLOWED_TEST_IDS", "").split(",") if x.strip()]

import time

_spotio_token_cache = {"token": None, "fetched_at": 0}
SPOTIO_TOKEN_TTL_SECONDS = 45 * 60


def get_spotio_token():
    now = time.time()
    if _spotio_token_cache["token"] and (now - _spotio_token_cache["fetched_at"] < SPOTIO_TOKEN_TTL_SECONDS):
        return _spotio_token_cache["token"]
    response = requests.post(
        f"{SPOTIO_BASE}/api/users/apitoken",
        headers={"Accept": "text/plain", "Content-Type": "application/merge-patch+json"},
        json={"clientId": SPOTIO_CLIENT_ID, "secret": SPOTIO_CLIENT_SECRET}
    )
    token = response.json()["accessToken"]
    _spotio_token_cache["token"] = token
    _spotio_token_cache["fetched_at"] = now
    return token


def transcribe_audio(url):
    response = requests.post(
        "https://api.deepgram.com/v1/listen",
        headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        json={"url": url}
    )
    data = response.json()
    try:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception:
        return f"Transcription failed: {data}"


XAI_API_KEY = os.environ.get("XAI_API_KEY")


def ask_grok(question):
    response = requests.post(
        "https://api.x.ai/v1/responses",
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "grok-4.3",
            "input": [{"role": "user", "content": question}],
            "tools": [{"type": "web_search"}]
        }
    )
    print(f"DEBUG: Grok response status {response.status_code}, body: {response.text[:800]}")

    if response.status_code != 200:
        return f"Grok call failed: {response.status_code} {response.text[:300]}"

    data = response.json()
    if data.get("output_text"):
        return data["output_text"]

    try:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content_block in item.get("content", []):
                    if content_block.get("type") in ("output_text", "text"):
                        return content_block.get("text", "")
        return f"Grok call succeeded but no text found in response: {str(data)[:500]}"
    except Exception as e:
        return f"Grok call failed to parse: {repr(e)} | raw: {str(data)[:500]}"


def update_spotio_field(record_type, record_id, fields):
    token = get_spotio_token()
    path_segment = "activities" if record_type == "activity" else "leads"
    patch_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/merge-patch+json",
        "Accept": "text/plain"
    }
    patch_resp = requests.patch(
        f"{SPOTIO_BASE}/api/v2/{path_segment}/{record_id}",
        headers=patch_headers,
        json=fields
    )
    print(f"DEBUG: PATCH {record_type} {record_id} with {fields}: {patch_resp.status_code} {patch_resp.text[:1000]}")
    return f"Status: {patch_resp.status_code}\nBody: {patch_resp.text[:1000]}"


def update_activity_notes(activity_id, notes):
    token = get_spotio_token()
    patch_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/merge-patch+json",
        "Accept": "text/plain"
    }
    patch_resp = requests.patch(
        f"{SPOTIO_BASE}/api/v2/activities/{activity_id}",
        headers=patch_headers,
        json={"notes": notes}
    )
    print(f"DEBUG: PATCH activity {activity_id}: {patch_resp.status_code} {patch_resp.text[:1000]}")
    return f"Status: {patch_resp.status_code}\nBody: {patch_resp.text[:1000]}"


MAX_TOOL_RESULT_CHARS = 6000


def truncate_result(text, limit=MAX_TOOL_RESULT_CHARS):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters omitted]"


def spotio_api_call(method, path, body=None):
    token = get_spotio_token()
    url = f"{SPOTIO_BASE}{path}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    print(f"DEBUG: Spotio API call: {method} {url} body={body}")
    response = requests.request(method, url, headers=headers, json=body)
    print(f"DEBUG: Spotio API response: {response.status_code} {response.text[:1000]}")
    return f"Status: {response.status_code}\nBody: {response.text[:2000]}"


tools = [
    {"type": "web_search_20250305", "name": "web_search"},
    {"type": "web_fetch_20250910", "name": "web_fetch"},
    {
        "name": "transcribe_audio",
        "description": "Transcribe a call recording from a direct audio URL (mp3/wav)",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    },
    {
        "name": "ask_grok",
        "description": (
            "Ask Grok (xAI's LLM, with live web search enabled) a question and get its answer "
            "back as text. When asking about a specific fact about a real person (e.g. their age), "
            "explicitly require Grok to cite a source — unsourced guesses should be reported as "
            "'not found' instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"]
        }
    },
    {
        "name": "update_spotio_field",
        "description": (
            "Correct a field on a Spotio activity (e.g. the appointment 'date') or a lead "
            "(e.g. 'address'). Only use this when you've found a CLEAR, obvious mistake. "
            "record_type must be 'activity' or 'lead'. fields is an object with just the "
            "field(s) you're correcting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_type": {"type": "string"},
                "record_id": {"type": "string"},
                "fields": {"type": "object"}
            },
            "required": ["record_type", "record_id", "fields"]
        }
    },
    {
        "name": "update_activity_notes",
        "description": (
            "Update the notes field on a Spotio activity. Just pass the activity_id "
            "and your notes text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_id": {"type": "string"},
                "notes": {"type": "string"}
            },
            "required": ["activity_id", "notes"]
        }
    },
    {
        "name": "spotio_api_call",
        "description": (
            "Make a read-only (GET) request to the Spotio REST API for looking up additional "
            "info if needed. For updating activity notes use update_activity_notes instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "path": {"type": "string"},
                "body": {"type": "object"}
            },
            "required": ["method", "path"]
        }
    }
]


@app.post("/webhook")
async def handle_lead(request: Request):
    try:
        return await process_lead(request)
    except Exception as e:
        import traceback
        print(f"DEBUG: UNHANDLED EXCEPTION: {repr(e)}")
        print(traceback.format_exc())
        return {"status": "error", "detail": repr(e)}


async def process_lead(request: Request):
    payload = await request.json()
    events = payload.get("payload", [])
    data = events[0].get("data", {}) if events else {}
    data_object = data.get("dataObject", {})

    fields = data_object.get("fields", [])
    field_map = {f.get("title", ""): (f.get("values") or [""])[0] for f in fields}
    first_name = field_map.get("First Name", "")
    last_name = field_map.get("Last Name", "")

    address = data_object.get("address", {}).get("fullAddress", "")
    notes = data.get("notes", "")
    activity_id = data.get("id", "") or data.get("objectId", "")
    lead_id = data_object.get("objectId", "")
    appointment_date = data.get("date", "")
    source = field_map.get("Source", "")
    assigned_email = data_object.get("assignedUserEmail", "")

    # Derive the activity creation date (date portion only) for day-of-week calculations
    activity_created_date = data.get("date", "")[:10]

    if ALLOWED_TEST_IDS and activity_id not in ALLOWED_TEST_IDS and lead_id not in ALLOWED_TEST_IDS:
        print(f"DEBUG: SKIPPING — not in test allowlist. activity_id={activity_id}")
        return {"status": "skipped (not in test allowlist)"}

    ALLOWED_SOURCES = ["growthify", "treasured leads"]
    if source.strip().lower() not in ALLOWED_SOURCES:
        print(f"DEBUG: SKIPPING — lead source is '{source}', not in allowed sources. activity_id={activity_id}")
        return {"status": f"skipped (source is '{source}', not in allowed sources)"}

    ALLOWED_ASSIGNED_EMAILS = ["info@growthifylabs.com"]
    if source.strip().lower() == "growthify" and assigned_email.strip().lower() not in ALLOWED_ASSIGNED_EMAILS:
        print(f"DEBUG: SKIPPING — Growthify lead not assigned to Growthify email. activity_id={activity_id}")
        return {"status": f"skipped (Growthify lead not assigned to Growthify, assigned to '{assigned_email}')"}

    print(f"DEBUG: Waiting 5 minutes before processing activity {activity_id}...")
    await asyncio.sleep(300)

    token = get_spotio_token()
    refreshed = requests.get(
        f"{SPOTIO_BASE}/api/v2/activities/{activity_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    if refreshed.status_code != 200:
        print(f"DEBUG: Could not fetch activity {activity_id} after delay: {refreshed.status_code}")
        return {"status": f"error fetching activity after delay: {refreshed.status_code}"}

    refreshed_notes = refreshed.json().get("notes", "")

    if "AUTOMATED CALL REVIEW SUMMARY" in refreshed_notes or "CALL SUMMARY" in refreshed_notes:
        print(f"DEBUG: SKIPPING — activity {activity_id} already has AI-generated notes.")
        return {"status": "skipped (already processed)"}

    if "drive.google.com" not in refreshed_notes and "http" not in refreshed_notes:
        print(f"DEBUG: SKIPPING — no recording link in notes after delay. activity_id={activity_id}")
        return {"status": "skipped (no recording link in notes)"}

    notes = refreshed_notes
    print(f"DEBUG: Notes after delay: {notes[:200]}")

    prompt = f"""A Spotio activity was created/updated:

Customer Name: {first_name} {last_name}
Address on file: {address}
Appointment date/time on file: {appointment_date}
Activity created date: {activity_created_date}
Activity notes (may contain a link): {notes}
Activity ID: {activity_id}
Related Lead ID: {lead_id}

Your job:

1. TRANSCRIBE THE CALL
   If the activity notes contain a link, fetch it (web_fetch). If it points to an audio file
   (e.g. a Google Drive share link), figure out the direct download URL and use transcribe_audio
   to get the call transcript.

2. CHECK FOR MISTAKES
   Compare what's said in the call transcript against the address and appointment date/time
   on file above.
   - If the call clearly states a DIFFERENT address than what's on file, and it's an obvious,
     unambiguous mistake (not just you guessing), correct it using update_spotio_field with
     record_type="lead", record_id={lead_id}, fields={{"address": {{"fullAddress": "..."}}}}.
   - If the call clearly states a DIFFERENT appointment date/time than what's on file, and it's
     an obvious, unambiguous mistake, correct it using update_spotio_field with
     record_type="activity", record_id={activity_id}, fields={{"date": "..."}} (use ISO 8601
     format matching the original, e.g. 2026-06-25T14:00:00+00:00).

     CRITICAL TIMEZONE RULE: Spotio stores all dates in UTC. All appointments in this
     territory are Eastern Time (ET). During EDT (Mar-Nov): ET = UTC-4, so 10:00 AM ET =
     14:00 UTC. During EST (Nov-Mar): ET = UTC-5, so 10:00 AM ET = 15:00 UTC. ALWAYS
     convert what the customer/rep says on the call (Eastern) to UTC before comparing to
     the Spotio value. Example: call says "10 AM", Spotio shows 14:00 UTC — these MATCH
     (both are 10 AM ET). Do NOT correct this. Only correct if the Eastern times genuinely
     differ after converting both to the same timezone.

     CRITICAL DAY-OF-WEEK RULE: Never rely on your own mental calculation of what day of
     the week a specific date falls on — this is error-prone. If the call mentions a day
     name (e.g. "this Friday", "next Monday"), you MUST use web_search to confirm exactly
     which calendar date that corresponds to, using the activity created date ({activity_created_date})
     as your reference point for what "this week" means. Only after confirming the exact
     date via web search should you compare it to the Spotio value and decide if a
     correction is needed.

   - Only make a correction if you're confident it's a real mistake. If you make any
     correction, note exactly what was changed in the notes in plain human-readable format
     (e.g. "July 10th at 5:00 PM"), never ISO format.

3. EXTRACT THESE SPECIFIC DETAILS FROM THE CALL (answer briefly, simply):
   - Was solar mentioned? If so, how many times?
   - Is the customer expecting someone to come in person?
   - Does the customer plan to be home?
   - Are they expecting a call beforehand?
   - What does the customer think their average electric bill is?

4. RESEARCH THE CUSTOMER'S EMPLOYER
   Use web_search to try to match this customer's name (and location, to disambiguate) to a
   Facebook or LinkedIn profile, to determine who they work for / what company. Note what you
   find (or that nothing reliable was found).

5. RESEARCH THE HOME SALE HISTORY
   Use web_search to find out when this home was last sold and for how much (e.g. via Zillow,
   Redfin, or county property records). Note what you find (or nothing reliable was found).

6. ASK GROK FOR THE CUSTOMER'S AGE
   Use ask_grok, giving it the customer's name and location, and ask it to find the customer's
   current age. Require it to cite a source for any age/birth year — no guessing. If it can't
   find a sourced answer, note "no verifiable age found."

7. WRITE A CLEAN, STRUCTURED SUMMARY using EXACTLY this format and structure:

Corrections made:
- [One short sentence: either "No corrections were necessary." followed by one sentence confirming address and appointment were verified, OR describe what was changed in plain language.]

Call details:
- Solar mentioned: [Yes/No and count, very brief]
- In person visit expected: [Yes/No]
- Does the customer plan to be home: [Answer, include relevant detail if they won't be home e.g. who will be there instead]
- Confirmation call expected: [Yes/No]
- Average electric bill: [Dollar amount or range only, no extra explanation]

Customer information:
- Employment: [One short phrase, or "Not found"]
- Age estimate: [Either "Unverified" if Grok found nothing, or the estimate with source in parentheses]
- Social media presence: [e.g. "None found" or "LinkedIn: [profile name/title]"]
- Home sale history: [Key property facts in one sentence, then sale price/date if found. 2-3 sentences max.]

   STRICT FORMATTING RULES:
   - Plain text only. No emojis, no arrows, no decorative symbols.
   - No divider lines (no "====", "----", "***") and no banner headers.
   - Every bullet starts with a simple dash and a space.
   - Dates/times in plain language only (e.g. "July 6th at 5:00 PM"), never ISO format.
   - Keep answers short. No source citations, no methodology, no "I searched..." language.
   - If something was not found, just say "Not found" or "Unverified."

8. Update the ACTIVITY's notes field (NOT the lead) with this summary, using
   update_activity_notes with activity_id={activity_id}.

9. Confirm the update succeeded by checking the response status code is 200.
"""

    print(f"DEBUG: raw payload: {payload}")
    print(f"DEBUG: prompt sent to Claude: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4096, tools=tools, messages=messages
    )
    print(f"DEBUG: Claude stop_reason: {response.stop_reason}")
    for block in response.content:
        if block.type == "text":
            print(f"DEBUG: Claude said: {block.text}")

    loop_count = 0
    while response.stop_reason == "tool_use" and loop_count < 15:
        loop_count += 1
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        tool_results = []
        for tool_use in tool_use_blocks:
            print(f"DEBUG: Claude called tool: {tool_use.name} with input: {tool_use.input}")

            if tool_use.name == "transcribe_audio":
                result = transcribe_audio(tool_use.input["url"])
                result_limit = None
            elif tool_use.name == "ask_grok":
                result = ask_grok(tool_use.input["question"])
                result_limit = 3000
            elif tool_use.name == "update_spotio_field":
                result = update_spotio_field(
                    tool_use.input["record_type"],
                    tool_use.input["record_id"],
                    tool_use.input["fields"]
                )
                result_limit = 1000
            elif tool_use.name == "update_activity_notes":
                result = update_activity_notes(tool_use.input["activity_id"], tool_use.input["notes"])
                result_limit = 1000
            elif tool_use.name == "spotio_api_call":
                result = spotio_api_call(
                    tool_use.input["method"],
                    tool_use.input["path"],
                    tool_use.input.get("body")
                )
                result_limit = 1500
            else:
                result = "handled automatically"
                result_limit = MAX_TOOL_RESULT_CHARS

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result) if result_limit is None else truncate_result(result, result_limit)
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096, tools=tools, messages=messages
        )
        print(f"DEBUG: Claude stop_reason: {response.stop_reason}")
        for block in response.content:
            if block.type == "text":
                print(f"DEBUG: Claude said: {block.text}")

    return {"status": "done"}


@app.get("/")
async def health_check():
    return {"status": "alive"}
