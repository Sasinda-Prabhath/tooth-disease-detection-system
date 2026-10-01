"""Verify the gateway preserves the workflow stream and session capability."""
import importlib.util
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_gateway_stream_forwarding(monkeypatch):
    path = Path(__file__).resolve().parents[3]/'api-gateway/app/routes/module2.py'
    spec = importlib.util.spec_from_file_location('workflow_gateway', path)
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    app = FastAPI()
    app.include_router(gateway.router)
    received = []
    events = b'{"type":"stage","prediction":{}}\n{"type":"result","prediction":{}}\n'

    async def upstream(request):
        received.append(request)
        assert request.headers['X-Case-Session'] == 'a'*64
        assert request.url.path == '/module2/workflow'
        body = await request.aread()
        assert b'opg.png' in body and b'pixel_spacing_mm' in body
        return httpx.Response(200, content=events, headers={'content-type': 'application/x-ndjson'})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(gateway.httpx, 'AsyncClient', lambda **kwargs: client_type(transport=httpx.MockTransport(upstream), **kwargs))
    with TestClient(app) as client:
        assert client.post('/module2/workflow', files={'file': ('opg.png', b'pixels')}).status_code == 401
        response = client.post('/module2/workflow', files={'file': ('opg.png', b'pixels')},
                               data={'pixel_spacing_mm': '.1'}, headers={'X-Case-Session': 'a'*64})
    assert len(received) == 1
    assert response.status_code == 200 and response.content == events
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-accel-buffering'] == 'no'
