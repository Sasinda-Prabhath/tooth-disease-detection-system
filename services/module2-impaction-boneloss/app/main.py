from contextlib import asynccontextmanager
import asyncio
import json
import logging
import os
import re
import threading
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from app.pipeline import Pipeline
from app.storage import Storage
from app.schemas import Prediction

@asynccontextmanager
async def lifespan(app):
    app.state.storage = Storage()
    app.state.storage.cleanup()
    app.state.pipeline = await run_in_threadpool(Pipeline, app.state.storage)
    async def cleanup():
        while True:
            await asyncio.sleep(60)
            try:
                await run_in_threadpool(app.state.storage.cleanup)
            except Exception:
                logging.getLogger(__name__).error('Case retention cleanup failed.')
    task = asyncio.create_task(cleanup())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    app.state.storage.engine.dispose()

app = FastAPI(title='Module 2: OPG research pipeline', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(','),
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'X-Case-Session'])

def session_id(value):
    if value is None or not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', value):
        raise HTTPException(401, 'A temporary X-Case-Session capability is required.')
    return value

@app.get('/health', include_in_schema=False)
@app.get('/module2/health')
def health():
    return app.state.pipeline.health()

@app.post('/predict', response_model=Prediction, include_in_schema=False)
@app.post('/module2/predict', response_model=Prediction)
async def predict(file: UploadFile = File(...), pixel_spacing_mm: float | None = Form(None, gt=0),
                  x_case_session: str | None = Header(None)):
    session = session_id(x_case_session)
    try:
        payload = await file.read(20*1024*1024+1)
    finally:
        await file.close()
    if len(payload) > 20*1024*1024:
        raise HTTPException(413, 'Maximum upload size is 20 MiB.')
    try:
        return await run_in_threadpool(app.state.pipeline.run, payload, file.filename or 'upload', pixel_spacing_mm, session)
    except ValueError as exc:
        raise HTTPException(422, 'Invalid or unsupported OPG: '+str(exc)) from exc

@app.get('/module2/cases/{case_id}/{kind}')
def artifact(case_id: str, kind: str, x_case_session: str | None = Header(None)):
    session = session_id(x_case_session)
    kinds = {'raw', 'processed', 'fdi-preview', 'impaction-preview', 'bone-loss-preview', 'final-image', 'final-report'} | {f'gradcam-{n}' for n in ('18', '28', '38', '48')}
    if not re.fullmatch(r'case_\d{8}_[a-f0-9]{16}', case_id) or kind not in kinds or not app.state.storage.authorised(case_id, session):
        raise HTTPException(404, 'Case artifact not found or expired.')
    try:
        data = app.state.storage.get(case_id, kind)
    except Exception:
        raise HTTPException(404, 'Case artifact unavailable.')
    return Response(data, media_type='application/json' if kind == 'final-report' else 'image/png',
                    headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer'})


@app.post('/module2/workflow')
async def workflow(file: UploadFile = File(...), pixel_spacing_mm: float | None = Form(None, gt=0),
                   x_case_session: str | None = Header(None)):
    session = session_id(x_case_session)
    try:
        payload = await file.read(20*1024*1024+1)
    finally:
        await file.close()
    if len(payload) > 20*1024*1024:
        raise HTTPException(413, 'Maximum upload size is 20 MiB.')

    async def events():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        disconnected = threading.Event()

        def progress(snapshot):
            if not disconnected.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, {'type': 'stage', 'prediction': snapshot})

        async def produce():
            try:
                result = await run_in_threadpool(app.state.pipeline.run, payload, file.filename or 'upload',
                                                pixel_spacing_mm, session, progress)
                await queue.put({'type': 'result', 'prediction': result.model_dump()})
            except ValueError as exc:
                await queue.put({'type': 'error', 'detail': 'Invalid or unsupported OPG: '+str(exc)})
            except Exception:
                logging.getLogger(__name__).exception('Workflow failed.')
                await queue.put({'type': 'error', 'detail': 'Analysis failed; no complete report is available.'})

        task = asyncio.create_task(produce())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, allow_nan=False)+'\n'
                if event['type'] in ('result', 'error'):
                    break
        finally:
            disconnected.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(events(), media_type='application/x-ndjson',
                             headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
