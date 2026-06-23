from fastapi import FastAPI, Request
import anthropic
import requests
import os

app = FastAPI()
client = anthropic.Anthropic()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
SPOTIO_CLIENT_ID = os.environ.get("SPOTIO_CLIENT_ID")
SPOTIO_CLIENT_SECRET = os.environ.get("SPOTIO_CLIENT_SECRET")


def get_spotio_token():
    response = requests.post(
        "https://api.spotio2.com/api/users/apitoken",
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


def update_spotio_notes(lead_id, notes):
    token = get_spotio_token()
    print(f"DEBUG: updating lead {lead_id} with notes: {notes[:200]}")
    response = requests.put(
        f"https://api.spotio2.com/api/leads/{lead_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"notes": notes}
    )
    print(f"DEBUG: Spotio response status {response.status_code}, body: {response.text[:300]}")
    return f"Status: {response.status_code}"


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
        "name": "update_spotio_notes",
        "description": "Update the notes field on a Spotio lead",
        "input_schema": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
                "notes": {"type": "string"}
            },
            "required": ["lead_id", "notes"]
        }
    }
]


@app.post("/webhook")
async def handle_lead(request: Request):
    payload = await request.json()
    lead = payload.get("lead", {})

    contacts = lead.get("contacts", [{}])
    first_name = contacts[0].get("firstName", "") if contacts else ""
    last_name = contacts[0].get("lastName", "") if contacts else ""
    address = lead.get("address", {}).get("fullAddress", "")
    notes = lead.get("notes", "")
    lead_id = lead.get("id", "")

    prompt = f"""New Spotio lead created:
Name: {first_name} {last_name}
Address: {address}
Notes field (may contain a link): {notes}
Lead ID: {lead_id}

Research the address and customer online. If the notes contain a link, fetch it.
If there's a call recording (direct audio URL), transcribe it and extract key details
(pain points, objections, next steps, sentiment). Write a clean, structured summary
and call update_spotio_notes with lead_id={lead_id}.
"""

    print(f"DEBUG: raw payload: {payload}")
    print(f"DEBUG: prompt sent to Claude: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2048, tools=tools, messages=messages
    )
    print(f"DEBUG: Claude stop_reason: {response.stop_reason}")
    for block in response.content:
        if block.type == "text":
            print(f"DEBUG: Claude said: {block.text}")

    while response.stop_reason == "tool_use":
        tool_use = next(b for b in response.content if b.type == "tool_use")
        print(f"DEBUG: Claude called tool: {tool_use.name} with input: {tool_use.input}")

        if tool_use.name == "transcribe_audio":
            result = transcribe_audio(tool_use.input["url"])
        elif tool_use.name == "update_spotio_notes":
            result = update_spotio_notes(tool_use.input["lead_id"], tool_use.input["notes"])
        else:
            result = "handled automatically"

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": str(result)}]
        })
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2048, tools=tools, messages=messages
        )
        print(f"DEBUG: Claude stop_reason: {response.stop_reason}")
        for block in response.content:
            if block.type == "text":
                print(f"DEBUG: Claude said: {block.text}")

    return {"status": "done"}


@app.get("/")
async def health_check():
    return {"status": "alive"}
