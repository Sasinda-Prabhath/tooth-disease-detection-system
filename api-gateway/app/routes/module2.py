"""Module 2 proxy preserving temporary case capabilities and private responses."""
import os
import re
import httpx
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask
router = APIRouter(prefix='/module2', tags=['module2'])
MODULE2_URL = os.getenv('MODULE2_URL', 'http://module2-service:8002')

def session_header(value):
    if value is None or not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', value):
        raise HTTPException(401, 'X-Case-Session is required.')
    return {'X-Case-Session': value}

async def forward(method, path, **kwargs):
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.request(method, MODULE2_URL+path, **kwargs)
        return Response(response.content, status_code=response.status_code, media_type=response.headers.get('content-type'), headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
    except httpx.RequestError:
        raise HTTPException(503, 'Module 2 service unavailable.')

@router.post('/predict')
async def predict(file: UploadFile = File(...), pixel_spacing_mm: float | None = Form(None, gt=0), x_case_session: str | None = Header(None)):
    headers = session_header(x_case_session)
    try:
        payload = await file.read(20*1024*1024+1)
    finally:
        await file.close()
    if len(payload) > 20*1024*1024:
        raise HTTPException(413, 'Maximum upload size is 20 MiB.')
    return await forward('POST', '/module2/predict', files={'file': (file.filename, payload, file.content_type)}, data={'pixel_spacing_mm': str(pixel_spacing_mm)} if pixel_spacing_mm is not None else {}, headers=headers)

@router.get('/health')
async def health():
    return await forward('GET', '/module2/health')


@router.post('/workflow')
async def workflow(file: UploadFile = File(...), pixel_spacing_mm: float | None = Form(None, gt=0), x_case_session: str | None = Header(None)):
    headers = session_header(x_case_session)
    try:
        payload = await file.read(20*1024*1024+1)
    finally:
        await file.close()
    if len(payload) > 20*1024*1024:
        raise HTTPException(413, 'Maximum upload size is 20 MiB.')
    client = httpx.AsyncClient(timeout=180)
    try:
        request = client.build_request('POST', MODULE2_URL+'/module2/workflow',
            files={'file': (file.filename, payload, file.content_type)},
            data={'pixel_spacing_mm': str(pixel_spacing_mm)} if pixel_spacing_mm is not None else {}, headers=headers)
        response = await client.send(request, stream=True)
    except httpx.RequestError:
        await client.aclose()
        raise HTTPException(503, 'Module 2 service unavailable.')

    async def close():
        await response.aclose()
        await client.aclose()

    return StreamingResponse(response.aiter_bytes(), status_code=response.status_code,
        media_type=response.headers.get('content-type'), background=BackgroundTask(close),
        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no', 'X-Content-Type-Options': 'nosniff'})

@router.get('/cases/{case_id}/{kind}')
async def artifact(case_id: str, kind: str, x_case_session: str | None = Header(None)):
    if not re.fullmatch(r'case_\d{8}_[a-f0-9]{16}', case_id) or not re.fullmatch(r'[a-z0-9-]+', kind):
        raise HTTPException(404, 'Artifact not found.')
    return await forward('GET', f'/module2/cases/{case_id}/{kind}', headers=session_header(x_case_session))
