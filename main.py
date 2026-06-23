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
        "name": "spotio_api_call",
        "description": (
            "Make a request to the Spotio REST API (base URL https://api.spotio2.com). "
            "Use this to read or update activities, leads, or any other Spotio resource. "
            "If you're unsure of the exact endpoint or required fields, use web_fetch to check "
            "the Spotio API docs/swagger at https://developer.spotio2.com first. "
            "method should be GET, PUT, POST, or PATCH. path should start with /api/... "
            "body is a JSON object for PUT/POST/PATCH requests (omit for GET)."
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
5. Update the ACTIVITY's notes field (NOT the lead) with this summary, using spotio_api_call.
   You are updating activity ID {activity_id}. If you're not sure of the exact endpoint path or
   what fields are required, use web_fetch to check https://developer.spotio2.com first
   (e.g. look for an "Activities" section, PUT/PATCH endpoint, and example request body).
   Make sure not to overwrite other required fields when you submit your update — fetch the
   current activity first with a GET request if needed, then include any required fields
   (like stageId, type, date, duration) alongside the updated notes field.
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
