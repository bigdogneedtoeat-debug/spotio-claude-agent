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


def update_activity_notes(activity_id, notes):
    token = get_spotio_token()
    headers = {"Authorization": f"Bearer {token}"}

    get_resp = requests.get(f"{SPOTIO_BASE}/api/v2/activities/{activity_id}", headers=headers)
    print(f"DEBUG: GET activity {activity_id}: {get_resp.status_code} {get_resp.text[:1000]}")
    if get_resp.status_code != 200:
        return f"Failed to fetch activity first: {get_resp.status_code} {get_resp.text[:300]}"

    activity = get_resp.json()

    # Set the new notes, preserving every other field/type exactly as Spotio returned it
    activity["notes"] = notes

    # Fields known to cause conflicts on PUT (computed/read-only or trigger dataObject edit checks)
    for field in ["dataObjectId", "dataObjectLat", "dataObjectLng", "activityLat", "activityLng",
                  "transition", "isNote", "autoExecution", "id"]:
        activity.pop(field, None)

    put_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    put_resp = requests.put(
        f"{SPOTIO_BASE}/api/v2/activities/{activity_id}",
        headers=put_headers,
        json=activity
    )
    print(f"DEBUG: PUT activity {activity_id}: {put_resp.status_code} {put_resp.text[:1000]}")

    return f"Status: {put_resp.status_code}\nBody: {put_resp.text[:1000]}"


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

    if ALLOWED_TEST_IDS and activity_id not in ALLOWED_TEST_IDS and lead_id not in ALLOWED_TEST_IDS:
        print(f"DEBUG: SKIPPING — activity_id={activity_id}, lead_id={lead_id} not in test allowlist {ALLOWED_TEST_IDS}")
        return {"status": "skipped (not in test allowlist)"}

    prompt = f"""A Spotio activity was created/updated:

Customer Name: {first_name} {last_name}
Address: {address}
Activity notes (may contain a link): {notes}
Activity ID: {activity_id}
Related Lead ID: {lead_id}

Your job:
1. Research the address and customer online (web_search).
2. If the activity notes contain a link, fetch it (web_fetch). If it points to an audio file
   (e.g. a Google Drive share link), figure out the direct download URL and use transcribe_audio
   to get the call transcript.
3. Extract key details from the call: pain points, objections, next steps, sentiment.
4. Write a clean, structured summary.
5. Update the ACTIVITY's notes field (NOT the lead) with this summary, using
   update_activity_notes with activity_id={activity_id}. This tool handles fetching the
   current activity and preserving its other fields for you — just pass the activity_id
   and your notes text.
6. Confirm the update succeeded by checking the response status code is 200.
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
