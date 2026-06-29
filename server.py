from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

app = FastAPI()

# Enable CORS to allow fetches from local browser files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_consumer_clients: Set[WebSocket] = set()

# Pre-defined daily conversational, emergency, and system-control sentences
SEMANTIC_CORPUS = [
    "আমাকে একটু সাহায্য করুন",
    "আমি শ্বাস নিতে পারছি না",
    "আমার খুব পানির পিপাসা লেগেছে",
    "আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো",
    "আমি বাথরুমে যেতে চাই",
    "আমি ঠিক আছি, চিন্তা করবেন না",
    "আপনাকে অনেক ধন্যবাদ",
    "দয়া করে ঘরের ফ্যানটি চালু করুন",
    "দয়া করে ঘরের লাইট বন্ধ করে দিন",
    "আমার চশমাটি কোথায় রেখেছেন?",
    "আমার এখন একটু ঘুম পাচ্ছে",
    "ডাক্তার ডাকুন দ্রুত",
    "কেমন আছেন সবাই?",
    "আমি ভালো আছি",
    "দয়া করে ঘরের দরজাটি খুলে দিন",
    "আমার শরীর ভালো লাগছে না"
]

# Initialize SentenceTransformer and Qdrant in-memory client
print("Loading LaBSE dense encoder model...", flush=True)
encoder = SentenceTransformer("sentence-transformers/LaBSE")

print("Initializing in-memory Qdrant database...", flush=True)
qdrant = QdrantClient(":memory:")
collection_name = "drishti_semantic"

qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Encode and index the corpus
print("Encoding and seeding semantic corpus...", flush=True)
embeddings = encoder.encode(SEMANTIC_CORPUS)
points = [
    PointStruct(
        id=idx,
        vector=emb.tolist(),
        payload={"text": text}
    )
    for idx, (text, emb) in enumerate(zip(SEMANTIC_CORPUS, embeddings))
]
qdrant.upsert(collection_name=collection_name, points=points)
print("Semantic corpus successfully seeded in-memory!", flush=True)


def get_time_slice() -> str:
    hour = datetime.now().hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


# Contextual Hashing Cache Layer
LEXICON_CACHE = {
    "morning": {
        "আ": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি ভালো আছি", "আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো"],
        "আম": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি ভালো আছি", "আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো"],
        "আমি": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি ভালো আছি", "আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো"],
        "আমির": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি ভালো আছি", "আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো"],
        "আমা": ["আমাকে একটু সাহায্য করুন", "আমার শরীর ভালো লাগছে না", "আমার খুব পানির পিপাসা লেগেছে"],
        "আমাকে": ["আমাকে একটু সাহায্য করুন", "আমার শরীর ভালো লাগছে না", "আমার খুব পানির পিপাসা লেগেছে"],
        "কে": ["কেমন আছেন সবাই?"],
        "কেম": ["কেমন আছেন সবাই?"],
        "কেমন": ["কেমন আছেন সবাই?"],
        "দ": ["দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন", "দয়া করে ঘরের লাইট বন্ধ করে দিন"],
        "দয়": ["দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন", "দয়া করে ঘরের লাইট বন্ধ করে দিন"],
        "দয়া": ["দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন", "দয়া করে ঘরের লাইট বন্ধ করে দিন"]
    },
    "afternoon": {
        "আ": ["আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো", "আমি বাথরুমে যেতে চাই", "আমি ঠিক আছি, চিন্তা করবেন না"],
        "আম": ["আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো", "আমি বাথরুমে যেতে চাই", "আমি ঠিক আছি, চিন্তা করবেন না"],
        "আমি": ["আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো", "আমি বাথরুমে যেতে চাই", "আমি ঠিক আছি, চিন্তা করবেন না"],
        "আমির": ["আমার ক্ষুধা লেগেছে, কিছু খাবার খাবো", "আমি বাথরুমে যেতে চাই", "আমি ঠিক আছি, চিন্তা করবেন না"],
        "আমা": ["আমাকে একটু সাহায্য করুন", "আমার খুব পানির পিপাসা লেগেছে", "আমার চশমাটি কোথায় রেখেছেন?"],
        "আমাকে": ["আমাকে একটু সাহায্য করুন", "আমার খুব পানির পিপাসা লেগেছে", "আমার চশমাটি কোথায় রেখেছেন?"],
        "কে": ["কেমন আছেন সবাই?"],
        "কেম": ["কেমন আছেন সবাই?"],
        "কেমন": ["কেমন আছেন সবাই?"],
        "দ": ["দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন", "দয়া করে ঘরের লাইট বন্ধ করে দিন"],
        "দয়": ["দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন", "দয়া করে ঘরের লাইট বন্ধ করে দিন"],
        "দয়া": ["দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন", "দয়া করে ঘরের লাইট বন্ধ করে দিন"]
    },
    "evening": {
        "আ": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি বাথরুমে যেতে চাই", "আমি ভালো আছি"],
        "আম": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি বাথরুমে যেতে চাই", "আমি ভালো আছি"],
        "আমি": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি বাথরুমে যেতে চাই", "আমি ভালো আছি"],
        "আমির": ["আমি ঠিক আছি, চিন্তা করবেন না", "আমি বাথরুমে যেতে চাই", "আমি ভালো আছি"],
        "আমা": ["আমাকে একটু সাহায্য করুন", "আমার খুব পানির পিপাসা লেগেছে", "আমার চশমাটি কোথায় রেখেছেন?"],
        "আমাকে": ["আমাকে একটু সাহায্য করুন", "আমার খুব পানির পিপাসা লেগেছে", "আমার চশমাটি কোথায় রেখেছেন?"],
        "দ": ["দয়া করে ঘরের লাইট বন্ধ করে দিন", "দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন"],
        "দয়": ["দয়া করে ঘরের লাইট বন্ধ করে দিন", "দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন"],
        "দয়া": ["দয়া করে ঘরের লাইট বন্ধ করে দিন", "দয়া করে ঘরের ফ্যানটি চালু করুন", "দয়া করে ঘরের দরজাটি খুলে দিন"],
        "আপ": ["আপনাকে অনেক ধন্যবাদ"],
        "আপন": ["আপনাকে অনেক ধন্যবাদ"],
        "আপনা": ["আপনাকে অনেক ধন্যবাদ"],
        "আপনাকে": ["আপনাকে অনেক ধন্যবাদ"]
    },
    "night": {
        "আ": ["আমার এখন একটু ঘুম পাচ্ছে", "আমি ঠিক আছি, চিন্তা করবেন না", "আমার শরীর ভালো লাগছে না"],
        "আম": ["আমার এখন একটু ঘুম পাচ্ছে", "আমি ঠিক আছি, চিন্তা করবেন না", "আমার শরীর ভালো লাগছে না"],
        "আমি": ["আমার এখন একটু ঘুম পাচ্ছে", "আমি ঠিক আছি, চিন্তা করবেন না", "আমার শরীর ভালো লাগছে না"],
        "আমির": ["আমার এখন একটু ঘুম পাচ্ছে", "আমি ঠিক আছি, চিন্তা করবেন না", "আমার শরীর ভালো লাগছে না"],
        "আমা": ["আমাকে একটু সাহায্য করুন", "আমার শরীর ভালো লাগছে না", "আমার চশমাটি কোথায় রেখেছেন?"],
        "আমাকে": ["আমাকে একটু সাহায্য করুন", "আমার শরীর ভালো লাগছে না", "আমার চশমাটি কোথায় রেখেছেন?"],
        "ডা": ["ডাক্তার ডাকুন দ্রুত"],
        "ডাক": ["ডাক্তার ডাকুন দ্রুত"],
        "ডাক্তার": ["ডাক্তার ডাকুন দ্রুত"],
        "দ": ["দয়া করে ঘরের লাইট বন্ধ করে দিন", "দয়া করে ঘরের দরজাটি খুলে দিন"],
        "দয়": ["দয়া করে ঘরের লাইট বন্ধ করে দিন", "দয়া করে ঘরের দরজাটি খুলে দিন"],
        "দয়া": ["দয়া করে ঘরের লাইট বন্ধ করে দিন", "দয়া করে ঘরের দরজাটি খুলে দিন"]
    }
}


class QueryRequest(BaseModel):
    text: str


@app.post("/api/semantic-complete")
async def semantic_complete(request: QueryRequest) -> dict:
    query_text = request.text.strip()
    if not query_text:
        return {"suggestions": []}
    
    time_slice = get_time_slice()
    
    # Check Predictive Cache Ring
    if time_slice in LEXICON_CACHE and query_text in LEXICON_CACHE[time_slice]:
        suggestions = LEXICON_CACHE[time_slice][query_text]
        print(f"[CACHE HIT] Bypassed LaBSE/Qdrant. Query: '{query_text.encode('utf-8', errors='ignore')}', Time slice: '{time_slice}', Suggestions count: {len(suggestions)}", flush=True)
        return {"suggestions": suggestions}
    
    print(f"[CACHE MISS] Running LaBSE + Qdrant. Query: '{query_text.encode('utf-8', errors='ignore')}', Time slice: '{time_slice}'", flush=True)
    try:
        # Encode query using LaBSE
        query_vector = encoder.encode(query_text).tolist()
        
        # Search Qdrant collection for top 3 hits using query_points
        response = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3
        )
        
        suggestions = [point.payload["text"] for point in response.points]
        return {"suggestions": suggestions}
    except Exception as e:
        print(f"Error in semantic completion: {e}", flush=True)
        return {"suggestions": []}


async def _safe_send(client: WebSocket, message: str) -> None:
    try:
        await client.send_text(message)
    except Exception:
        _consumer_clients.discard(client)


async def _broadcast(message: str) -> None:
    for client in list(_consumer_clients):
        # Schedule in the background so slow consumers never block the event loop
        asyncio.create_task(_safe_send(client, message))


@app.get("/")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/consumer")
async def consumer_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    _consumer_clients.add(websocket)
    await websocket.send_text('{"type":"status","message":"consumer_connected"}')
    print(f"Consumer connected. total={len(_consumer_clients)}", flush=True)
    try:
        while True:
            # Active receive loop to detect browser refresh/disconnect instantly
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception) as e:
        print(f"Consumer disconnect or error: {e}", flush=True)
    finally:
        _consumer_clients.discard(websocket)
        print(f"Consumer disconnected. total={len(_consumer_clients)}", flush=True)


@app.websocket("/ws/producer")
async def producer_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    print("Producer connected", flush=True)
    try:
        while True:
            message = await websocket.receive_text()
            # Broadcast asynchronously to achieve ultra-low latency without blocking the loop
            await _broadcast(message)
    except WebSocketDisconnect:
        print("Producer disconnected", flush=True)
        return
    except Exception as e:
        print(f"Producer connection error: {e}", flush=True)
        return
