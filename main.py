from fastapi import FastAPI, Request
import anthropic
import requests
import os

app = FastAPI()
client = anthropic.Anthropic()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
SPOTIO_CLIENT_ID = os.environ.get("SPOTIO_CLIENT_ID")
SPOTIO_CLIENT_SECRET = os.environ.get("SPOTIO_CLIENT_SECRET")

SPOTIO_BASE = "https://api.spotio2.com"

# SAFETY: only process these lead/activity IDs while testing.
# Leave empty (i.e. "") in the ALLOWED_TEST_IDS env var to process everything (production mode).
ALLOWED_TEST_IDS = [x.strip() for x in os.environ.get("ALLOWED_TEST_IDS", "").split(",") if x.strip()]


def get_spotio_token():
    response = requests.post(
        f"{SPOTIO_BASE}/api/users/apitoken",
        headers={"Accept": "text/plain", "Content-Type": "application/merge-patch+json"},
        json={"clientId": SPOTIO_CLIENT_ID, "secret": SPOTIO_CLIENT_SECRET}
    )
    return response.json()["accessToken"]


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
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "grok-4",
            "messages": [{"role": "user", "content": question}]
        }
    )
    print(f"DEBUG: Grok response status {response.status_code}, body: {response.text[:500]}")
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return f"Grok call failed: {response.status_code} {response.text[:300]}"


def update_spotio_field(record_type, record_id, fields):
    """
    record_type: 'activity' or 'lead'
    fields: dict of field(s) to merge-patch in, e.g. {"date": "2026-06-25T14:00:00+00:00"}
            or {"address": {"fullAddress": "...", "street": "...", ...}}
    """
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


def spotio_api_call(method, path, body=None):
    """Generic Spotio API caller. path should start with /api/..."""
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
            "Ask Grok (xAI's LLM) a question and get its answer back as text. "
            "Useful for a second opinion, alternate perspective, or anything you want Grok's take on."
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
            "(e.g. 'address'). Only use this when you've found a CLEAR, obvious mistake — "
            "e.g. the call recording states a different appointment time or address than "
            "what's in Spotio. record_type must be 'activity' or 'lead'. fields is an object "
            "with just the field(s) you're correcting, e.g. {\"date\": \"2026-06-25T14:00:00+00:00\"} "
            "for an activity, or {\"address\": {\"fullAddress\": \"123 Main St, City, ST 00000\"}} for a lead."
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
            "Update the notes field on a Spotio activity. This automatically fetches the "
            "current activity first, preserves all its other required fields with correct "
            "data types, and writes back your new notes text. Use this instead of trying to "
            "construct the raw API request yourself."
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
            "Make a read-only (GET) request to the Spotio REST API (base URL https://api.spotio2.com) "
            "for looking up additional info if needed. For updating activity notes, use "
            "update_activity_notes instead — do not try to PUT/PATCH activities with this tool."
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

    if ALLOWED_TEST_IDS and activity_id not in ALLOWED_TEST_IDS and lead_id not in ALLOWED_TEST_IDS:
        print(f"DEBUG: SKIPPING — activity_id={activity_id}, lead_id={lead_id} not in test allowlist {ALLOWED_TEST_IDS}")
        return {"status": "skipped (not in test allowlist)"}

    prompt = f"""A Spotio activity was created/updated:

Customer Name: {first_name} {last_name}
Address on file: {address}
Appointment date/time on file: {appointment_date}
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
   - Only make a correction if you're confident it's a real mistake, not a guess. If you make
     any correction, note exactly what was changed (old value -> new value) so it can be
     included in the notes.

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
   Redfin, or county property records). Note what you find (or that nothing reliable was found).

6. ASK GROK FOR THE CUSTOMER'S AGE
   Use ask_grok, giving it the customer's name and location, and ask it to estimate/provide the
   customer's current age. Include whatever it answers in the notes (clearly labeled as Grok's
   answer, and treat it as an unverified estimate).

7. WRITE A CLEAN, STRUCTURED SUMMARY containing:
   - Any corrections made (old value -> new value), if any
   - The 5 quick call details from step 3
   - The employer research from step 4
   - The home sale history from step 5
   - The age estimate from Grok (step 6)
   Keep it concise and easy to scan — short bullet points, not long paragraphs.

8. Update the ACTIVITY's notes field (NOT the lead) with this summary, using
   update_activity_notes with activity_id={activity_id}. This tool handles fetching the
   current activity and preserving its other fields for you — just pass the activity_id
   and your notes text.

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
        tool_use = next(b for b in response.content if b.type == "tool_use")
        print(f"DEBUG: Claude called tool: {tool_use.name} with input: {tool_use.input}")

        if tool_use.name == "transcribe_audio":
            result = transcribe_audio(tool_use.input["url"])
        elif tool_use.name == "ask_grok":
            result = ask_grok(tool_use.input["question"])
        elif tool_use.name == "update_spotio_field":
            result = update_spotio_field(
                tool_use.input["record_type"],
                tool_use.input["record_id"],
                tool_use.input["fields"]
            )
        elif tool_use.name == "update_activity_notes":
            result = update_activity_notes(tool_use.input["activity_id"], tool_use.input["notes"])
        elif tool_use.name == "spotio_api_call":
            result = spotio_api_call(
                tool_use.input["method"],
                tool_use.input["path"],
                tool_use.input.get("body")
            )
        else:
            result = "handled automatically"

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": str(result)}]
        })
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
