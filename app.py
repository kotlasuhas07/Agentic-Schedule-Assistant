import os
import json
import uuid
import sqlite3
import random
from datetime import datetime, date, time, timedelta

import chromadb
from google import genai
from google.genai import types
import gradio as gr

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
CURRENT_DATE = date(2026, 8, 15)
CURRENT_DATE_STR = CURRENT_DATE.isoformat()
SQLITE_DB_PATH = "schedule.db"
CHROMA_DB_PATH = "./schedule_chroma_db"
CHROMA_COLLECTION_NAME = "schedule_events"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set the GOOGLE_API_KEY secret in your Space settings.")
client = genai.Client(api_key=GOOGLE_API_KEY)

EVENT_TYPES = ["Meeting", "Workshop", "Task", "Appointment", "Deadline", "Personal"]
random.seed(42)

def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("CREATE TABLE IF NOT EXISTS schedule ("
                 "id TEXT PRIMARY KEY, title TEXT NOT NULL, event_type TEXT NOT NULL, "
                 "date TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL, description TEXT)")
    conn.commit(); conn.close()

def db_insert_event(e):
    conn = get_db_connection()
    conn.execute("INSERT OR REPLACE INTO schedule VALUES (?,?,?,?,?,?,?)",
                 (e["id"], e["title"], e["event_type"], e["date"], e["start_time"], e["end_time"], e.get("description","")))
    conn.commit(); conn.close()

def db_update_event(event_id, updates):
    conn = get_db_connection()
    fields, values = [], []
    for k in ["title","event_type","date","start_time","end_time","description"]:
        if updates.get(k) is not None:
            fields.append(f"{k} = ?"); values.append(updates[k])
    if fields:
        values.append(event_id)
        conn.execute(f"UPDATE schedule SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    conn.close()

def db_delete_event(event_id):
    conn = get_db_connection(); conn.execute("DELETE FROM schedule WHERE id = ?", (event_id,)); conn.commit(); conn.close()

def db_get_all_events():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM schedule ORDER BY date, start_time").fetchall()
    conn.close(); return [dict(r) for r in rows]

def db_get_event_by_id(event_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM schedule WHERE id = ?", (event_id,)).fetchone()
    conn.close(); return dict(row) if row else None

def seed_sample_events():
    if db_get_all_events():
        return
    templates = {
        "Meeting": ("Team Meeting", "Weekly project discussion."),
        "Workshop": ("Python Workshop", "Hands-on Python workshop for the team."),
        "Task": ("Complete Project Report", "Finish writing the quarterly report."),
        "Appointment": ("Doctor Appointment", "Routine check-up."),
        "Deadline": ("Submit Assignment", "Final deadline to submit the assignment."),
        "Personal": ("Gym Session", "Personal workout session."),
    }
    for i in range(30):
        d = CURRENT_DATE + timedelta(days=i % 30)
        etype = EVENT_TYPES[i % len(EVENT_TYPES)]
        title, desc = templates[etype]
        start_h = 8 + (i % 10)
        db_insert_event({"id": str(uuid.uuid4()), "title": title, "event_type": etype,
                          "date": d.isoformat(), "start_time": f"{start_h:02d}:00",
                          "end_time": f"{start_h+1:02d}:00", "description": desc})

init_db(); seed_sample_events()

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

def get_embedding(text, task_type="RETRIEVAL_DOCUMENT"):
    r = client.models.embed_content(model=EMBEDDING_MODEL, contents=text,
                                     config=types.EmbedContentConfig(task_type=task_type))
    return r.embeddings[0].values

def event_to_text(e):
    d = datetime.strptime(e["date"], "%Y-%m-%d").strftime("%B %d, %Y")
    st = datetime.strptime(e["start_time"], "%H:%M").strftime("%I:%M %p").lstrip("0")
    et = datetime.strptime(e["end_time"], "%H:%M").strftime("%I:%M %p").lstrip("0")
    return f"{e['title']} | {e['event_type']} | {d} | {st} - {et} | {e.get('description', '')}"

def sync_event_to_chroma(e):
    text = event_to_text(e)
    emb = get_embedding(text, "RETRIEVAL_DOCUMENT")
    collection.upsert(ids=[e["id"]], embeddings=[emb], documents=[text],
                       metadatas={"date": e["date"], "event_type": e["event_type"]})

def remove_event_from_chroma(event_id):
    try: collection.delete(ids=[event_id])
    except Exception: pass

for e in db_get_all_events():
    sync_event_to_chroma(e)

def retrieve_schedule_context(query, n_results=5):
    emb = get_embedding(query, "RETRIEVAL_QUERY")
    res = collection.query(query_embeddings=[emb], n_results=n_results)
    return res.get("ids", [[]])[0]

def to_time(t): return datetime.strptime(t, "%H:%M").time()
def times_overlap(s1,e1,s2,e2):
    a,b,c,d = to_time(s1), to_time(e1), to_time(s2), to_time(e2)
    return a < d and c < b

def check_conflict(date_str, start_time, end_time, exclude_id=None):
    same_day = [e for e in db_get_all_events() if e["date"] == date_str]
    return [e for e in same_day if e["id"] != exclude_id and times_overlap(e["start_time"], e["end_time"], start_time, end_time)]

def get_schedule(date=None, date_range_start=None, date_range_end=None, start_time=None, end_time=None, query=None):
    events = db_get_all_events()
    results = events
    used = False
    if date:
        results = [e for e in results if e["date"] == date]; used = True
    elif date_range_start and date_range_end:
        results = [e for e in results if date_range_start <= e["date"] <= date_range_end]; used = True
    if start_time and end_time:
        results = [e for e in results if times_overlap(e["start_time"], e["end_time"], start_time, end_time)]; used = True
    if query:
        ids = set(retrieve_schedule_context(query, 10))
        if used:
            refined = [e for e in results if e["id"] in ids]
            results = refined if refined else results
        else:
            id_map = {e["id"]: e for e in events}
            results = [id_map[i] for i in ids if i in id_map]
    results = sorted(results, key=lambda e: (e["date"], e["start_time"]))
    return {"count": len(results), "events": results}

def update_schedule(action, event_id=None, title=None, event_type=None, date=None,
                     start_time=None, end_time=None, description=None, force=False):
    action = action.lower().strip()
    if action == "add":
        if not all([title, event_type, date, start_time, end_time]):
            return {"status": "error", "message": "Missing required fields."}
        if not force:
            c = check_conflict(date, start_time, end_time)
            if c: return {"status": "conflict", "conflicts": c}
        ev = {"id": str(uuid.uuid4()), "title": title, "event_type": event_type, "date": date,
              "start_time": start_time, "end_time": end_time, "description": description or ""}
        db_insert_event(ev); sync_event_to_chroma(ev)
        return {"status": "success", "event": ev}
    elif action == "update":
        if not event_id: return {"status": "error", "message": "event_id required."}
        existing = db_get_event_by_id(event_id)
        if not existing: return {"status": "error", "message": "Event not found."}
        updates = {k: v for k, v in {"title": title, "event_type": event_type, "date": date,
                   "start_time": start_time, "end_time": end_time, "description": description}.items() if v is not None}
        merged = {**existing, **updates}
        if not force and (date or start_time or end_time):
            c = check_conflict(merged["date"], merged["start_time"], merged["end_time"], event_id)
            if c: return {"status": "conflict", "conflicts": c}
        db_update_event(event_id, updates)
        updated = db_get_event_by_id(event_id)
        sync_event_to_chroma(updated)
        return {"status": "success", "event": updated}
    elif action == "delete":
        if not event_id: return {"status": "error", "message": "event_id required."}
        existing = db_get_event_by_id(event_id)
        if not existing: return {"status": "error", "message": "Event not found."}
        db_delete_event(event_id); remove_event_from_chroma(event_id)
        return {"status": "success", "event": existing}
    return {"status": "error", "message": "Unknown action."}

get_schedule_declaration = types.FunctionDeclaration(
    name="get_schedule", description="Retrieve schedule events by date, date range, time, and/or free-text query.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={
        "date": types.Schema(type=types.Type.STRING), "date_range_start": types.Schema(type=types.Type.STRING),
        "date_range_end": types.Schema(type=types.Type.STRING), "start_time": types.Schema(type=types.Type.STRING),
        "end_time": types.Schema(type=types.Type.STRING), "query": types.Schema(type=types.Type.STRING)}))

update_schedule_declaration = types.FunctionDeclaration(
    name="update_schedule", description="Add, update, or delete a schedule event.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={
        "action": types.Schema(type=types.Type.STRING), "event_id": types.Schema(type=types.Type.STRING),
        "title": types.Schema(type=types.Type.STRING), "event_type": types.Schema(type=types.Type.STRING),
        "date": types.Schema(type=types.Type.STRING), "start_time": types.Schema(type=types.Type.STRING),
        "end_time": types.Schema(type=types.Type.STRING), "description": types.Schema(type=types.Type.STRING)},
        required=["action"]))

schedule_tool = types.Tool(function_declarations=[get_schedule_declaration, update_schedule_declaration])

SYSTEM_INSTRUCTION = f"You are a helpful schedule assistant. Today is {CURRENT_DATE_STR}. Resolve relative dates/times to exact YYYY-MM-DD / HH:MM before calling tools. Afternoon=12:00-17:00, morning=08:00-12:00, evening=17:00-21:00. Use get_schedule for lookups/availability, update_schedule for add/update/delete (look up event_id via get_schedule first if needed). Explain conflicts clearly."

def call_tool(name, args):
    if name == "get_schedule": return get_schedule(**args)
    if name == "update_schedule": return update_schedule(**args)
    return {"status": "error", "message": "unknown tool"}

def run_agent(user_query):
    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, tools=[schedule_tool])
    contents = [types.Content(role="user", parts=[types.Part(text=user_query)])]
    response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
    fc_part = None
    for part in response.candidates[0].content.parts:
        if getattr(part, "function_call", None):
            fc_part = part; break
    if fc_part is None:
        return response.text
    fn_name = fc_part.function_call.name
    fn_args = dict(fc_part.function_call.args)
    tool_result = call_tool(fn_name, fn_args)
    contents.append(response.candidates[0].content)
    contents.append(types.Content(role="user", parts=[types.Part.from_function_response(name=fn_name, response={"result": tool_result})]))
    final = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
    return final.text

def chat_fn(message, history):
    try:
        return run_agent(message)
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Agentic RAG Schedule Assistant") as demo:
    gr.Markdown("# Agentic RAG Schedule Assistant")
    gr.ChatInterface(fn=chat_fn)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
