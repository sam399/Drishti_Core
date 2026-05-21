from __future__ import annotations

import asyncio
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

_consumer_clients: Set[WebSocket] = set()


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
