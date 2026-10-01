from io import BytesIO
import json
import time
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import Tooth, Site, ImpactionResult, BoneResult
from app.services.preprocessing import decode, preprocess
from app.services.fdi_detection import validate_teeth
from app.services.bone_loss import measure, landmarks_from_mask
from app.services.impaction import crop_and_geometry
from training.prepare import split_patients, check_splits

def payload():
    image = np.random.default_rng(42).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    stream = BytesIO(); Image.fromarray(image).save(stream, 'PNG')
    return stream.getvalue()

def tooth(fdi='48', confidence=.9, box=(30, 110, 80, 180)):
    x, y, xx, yy = box
    return Tooth(fdi=fdi, confidence=confidence, box_xyxy=box, polygon=[(x,y), (xx,y), (xx,yy), (x,yy)])

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('MODULE2_STORAGE_DIR', str(tmp_path/'objects'))
    monkeypatch.setenv('MODELS_DIR', str(tmp_path/'missing-models'))
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.delenv('S3_BUCKET', raising=False)
    with TestClient(app) as client:
        yield client

SESSION = {'X-Case-Session': 'a'*64}

def test_degraded_and_session_isolation(client):
    assert client.get('/module2/health').json()['status'] == 'degraded'
    assert client.post('/module2/predict', files={'file': ('opg.png', payload())}).status_code == 401
    response = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION)
    assert response.status_code == 200
    result = response.json()
    assert result['model_status'] == 'degraded'
    assert result['teeth'] == [] and result['third_molars'] == []
    assert result['processed_image_url'] is None
    assert result['pixel_spacing_mm'] is None
    raw = result['raw_image_url']
    assert client.get(raw).status_code == 401
    assert client.get(raw, headers={'X-Case-Session': 'b'*64}).status_code == 404
    response = client.get(raw, headers=SESSION)
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert client.get(raw.replace('/raw', '/unknown'), headers=SESSION).status_code == 404

def test_expired_case_deleted(client):
    from sqlalchemy.orm import Session
    from app.storage import Case
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION).json()
    storage = app.state.storage
    with Session(storage.engine) as db:
        case = db.get(Case, result['case_id']); case.expires = time.time()-1; db.commit()
    assert client.get(result['raw_image_url'], headers=SESSION).status_code == 404
    storage.cleanup()
    assert not (storage.raw_root/f'{result["case_id"]}.png').exists()

@pytest.mark.parametrize('spacing', ['-1', '0', 'nan', 'inf'])
def test_bad_calibration(client, spacing):
    response = client.post('/module2/predict', files={'file': ('opg.png', payload())}, data={'pixel_spacing_mm': spacing}, headers=SESSION)
    assert response.status_code == 422

def test_quality_and_invalid_image(client):
    assert client.post('/module2/predict', files={'file': ('opg.png', b'bad')}, headers=SESSION).status_code == 422
    assert client.post('/module2/predict', files={'file': ('opg.dcm', b'bad')}, headers=SESSION).status_code == 422
    stream = BytesIO(); Image.new('RGB', (400, 200), 'gray').save(stream, 'PNG')
    with pytest.raises(ValueError, match='quality'):
        decode(stream.getvalue(), 'opg.png')

def test_anisotropic_spacing_and_null_landmarks():
    sites = [Site(side='mesial', cej_point=(0, 0), crest_point=(3, 4)), Site(side='distal')]
    result = measure('48', sites, (2., 1.))
    assert result.mean_bone_loss_mm == pytest.approx(np.sqrt(73), abs=.0001)
    assert result.sites[1].cej_point is None and result.sites[1].bone_loss_mm is None
    assert result.severity == 'Not assessed'
    no_calibration = measure('48', [Site(side='mesial', cej_point=(0, 0), crest_point=(3, 4))], None)
    assert not no_calibration.assessable and no_calibration.mean_bone_loss_mm is None

def test_fdi_dedup_uncertainty_and_sequence():
    teeth = validate_teeth([tooth(), tooth(confidence=.2), tooth('47', .3, (90, 110, 130, 180))])
    assert len(teeth) == 2 and next(t for t in teeth if t.fdi == '48').confidence == .9
    assert next(t for t in teeth if t.fdi == '47').status == 'uncertain'
    wrong = validate_teeth([tooth('48', box=(90, 110, 130, 180)), tooth('47')])
    assert all(t.status == 'uncertain' for t in wrong)

def test_crop_includes_adjacent_and_clean_pixels():
    rgb, _, _ = decode(payload(), 'opg.png')
    t, adjacent = tooth(), tooth('47', box=(90, 110, 130, 180))
    crop, (x, y), geometry = crop_and_geometry(rgb, t, [t, adjacent])
    assert x <= 30 and x+crop.shape[1] >= 130
    np.testing.assert_array_equal(crop, rgb[y:y+crop.shape[0], x:x+crop.shape[1]])
    assert geometry.shape == (12,) and geometry[-1] == 1

def test_preprocessing_order_keeps_mask_black():
    class Privacy:
        def mask(self, rgb):
            mask = np.zeros(rgb.shape[:2], np.uint8); mask[:20] = 255
            return mask, []
    rgb, _, _ = decode(payload(), 'opg.png')
    processed, _ = preprocess(rgb, Privacy())
    assert processed.dtype == np.uint8 and processed.shape == rgb.shape
    assert processed[:20].max() == 0

def test_patient_splits_repeatable_and_leakage_rejected():
    rows = [{'patient_id': str(i//2), 'image': f'{i}.png'} for i in range(20)]
    split = split_patients(rows)
    assert split == split_patients(rows)
    check_splits(split)
    split[1]['split'] = 'test' if split[0]['split'] != 'test' else 'train'
    with pytest.raises(ValueError, match='leakage'):
        check_splits(split)

def test_loaded_pipeline_uses_clean_inputs_and_report_excludes_raw(client):
    # Explicit test doubles verify orchestration, never model accuracy.
    observed = {}
    class Privacy:
        def mask(self, rgb): return np.zeros(rgb.shape[:2], np.uint8), []
    class FDI:
        def predict(self, rgb): observed['processed'] = rgb.copy(); return [tooth()]
    class Classifier:
        def predict(self, crop, features, fdi):
            observed['crop'] = crop.copy()
            return ImpactionResult(fdi=fdi, impacted=False, impaction_confidence=.95)
        def gradcam(self, crop, features): return crop
    class Bone:
        def predict(self, crop, tooth, origin, spacing):
            np.testing.assert_array_equal(crop, observed['crop'])
            return measure(tooth.fdi, [Site(side='mesial', cej_point=(40, 120), crest_point=(40, 130))], spacing)
    pipeline = app.state.pipeline
    pipeline.models = {'privacy': Privacy(), 'fdi': FDI(), 'impaction': Classifier(), 'bone_loss': Bone()}
    pipeline.status = {name: 'loaded' for name in pipeline.models}
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION).json()
    assert result['model_status'] == 'degraded'  # Missing calibration makes bone assessment partial.
    assert result['third_molars'][0]['angulation'] is None
    assert not result['bone_loss_results'][0]['assessable']
    for key in ('processed_image_url', 'fdi_preview_url', 'impaction_preview_url', 'bone_loss_preview_url', 'final_image_url', 'final_report_url'):
        assert client.get(result[key], headers=SESSION).status_code == 200
    report = client.get(result['final_report_url'], headers=SESSION).json()
    assert 'raw_image_url' not in report
    assert '/raw' not in json.dumps(report)
    crop, _, _ = crop_and_geometry(observed['processed'], tooth(), [tooth()])
    np.testing.assert_array_equal(crop, observed['crop'])

def test_dicom_calibration_and_monochrome1():
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid(); meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
    ds.Rows = 160; ds.Columns = 320; ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = 'MONOCHROME1'; ds.PixelSpacing = [.2, .1]
    ds.PixelData = np.tile(np.arange(320, dtype=np.uint16), (160, 1)).tobytes()
    stream = BytesIO(); ds.save_as(stream, enforce_file_format=True)
    rgb, spacing, source = decode(stream.getvalue(), 'scan.dcm')
    assert spacing == (.2, .1) and 'DICOM' in source
    assert rgb[0, 0, 0] == 255 and rgb[0, -1, 0] == 0

def test_ambiguous_landmarks_abstain():
    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[120:124, 60:64] = 1; mask[140:144, 65:69] = 1
    sites = landmarks_from_mask(mask, tooth(), (0, 0))
    assert all(site.cej_point is None for site in sites)

def test_missing_spacing_clears_previous_measurement():
    sites = [Site(side='mesial', cej_point=(1, 1), crest_point=(2, 2))]
    assert measure('48', sites, (.1, .1)).sites[0].bone_loss_mm is not None
    assert measure('48', sites, None).sites[0].bone_loss_mm is None

def test_horizontal_molar_landmarks_abstain():
    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[120:124, 60:64] = 1; mask[130:134, 60:64] = 2
    sites = landmarks_from_mask(mask, tooth(box=(30, 110, 180, 150)), (0, 0))
    assert all(site.cej_point is None and site.crest_point is None for site in sites)

def test_internal_text_blocks_privacy_pipeline(client):
    class Privacy:
        def mask(self, rgb): raise ValueError('Text detected inside diagnostic region; manual review required.')
    app.state.pipeline.models['privacy'] = Privacy()
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION).json()
    assert result['processed_image_url'] is None and result['teeth'] == []
    assert any('manual review' in warning for warning in result['warnings'])


def test_blocked_workflow_still_produces_report(client):
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION).json()
    assert [s['status'] for s in result['workflow']] == ['completed', 'blocked', 'blocked', 'blocked', 'blocked', 'completed']
    report = client.get(result['final_report_url'], headers=SESSION)
    assert report.status_code == 200
    assert '/raw' not in report.text
    assert client.get(result['final_report_url'], headers={'X-Case-Session': 'b'*64}).status_code == 404


def install_workflow_models(fail_impaction=False):
    calls = []
    class Privacy:
        def mask(self, rgb): return np.zeros(rgb.shape[:2], np.uint8), []
    class FDI:
        def predict(self, rgb):
            calls.append('fdi')
            return [tooth(), tooth('38', box=(280, 110, 330, 180))]
    class Classifier:
        def predict(self, crop, features, fdi):
            calls.append('impaction-'+fdi)
            if fail_impaction: raise RuntimeError('test failure')
            return ImpactionResult(fdi=fdi, impacted=True, angulation='Mesioangular')
        def gradcam(self, crop, features): return crop
    class Bone:
        def predict(self, crop, tooth, origin, spacing):
            calls.append('bone-'+tooth.fdi)
            return measure(tooth.fdi, [Site(side='mesial', cej_point=(40, 120), crest_point=(40, 130))], spacing)
    pipeline = app.state.pipeline
    pipeline.models = {'privacy': Privacy(), 'fdi': FDI(), 'impaction': Classifier(), 'bone_loss': Bone()}
    pipeline.status = {name: 'loaded' for name in pipeline.models}
    return calls


def test_streamed_workflow_is_ordered_and_artifacts_are_protected(client):
    calls = install_workflow_models()
    response = client.post('/module2/workflow', files={'file': ('opg.png', payload())},
                           data={'pixel_spacing_mm': '.1'}, headers=SESSION)
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['type'] == 'result'
    result = events[-1]['prediction']
    assert calls == ['fdi', 'impaction-48', 'impaction-38', 'bone-48', 'bone-38']
    assert all(s['status'] == 'completed' for s in result['workflow'])
    assert result['model_status'] == 'ready'
    fdi_event = next(e['prediction'] for e in events if e['prediction']['workflow'][2]['status'] == 'completed')
    assert fdi_event['fdi_preview_url'] and not fdi_event['third_molars']
    impaction_event = next(e['prediction'] for e in events if e['prediction']['workflow'][3]['status'] == 'completed')
    assert impaction_event['impaction_preview_url'] and not impaction_event['bone_loss_results']
    for key in ('impaction_preview_url', 'bone_loss_preview_url'):
        assert client.get(result[key], headers=SESSION).status_code == 200
        assert client.get(result[key], headers={'X-Case-Session': 'b'*64}).status_code == 404
    assert client.get(result['final_report_url'], headers=SESSION).json()['workflow'][-1]['status'] == 'completed'


def test_impaction_failure_preserves_bone_assessment(client):
    install_workflow_models(fail_impaction=True)
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())},
                         data={'pixel_spacing_mm': '.1'}, headers=SESSION).json()
    assert result['workflow'][3]['status'] == 'partial'
    assert result['workflow'][4]['status'] == 'completed'
    assert all(t['impacted'] is None for t in result['third_molars'])
    assert all(b['assessable'] for b in result['bone_loss_results'])
    assert result['final_report_url']


def test_streamed_invalid_upload_and_missing_session(client):
    assert client.post('/module2/workflow', files={'file': ('opg.png', payload())}).status_code == 401
    response = client.post('/module2/workflow', files={'file': ('opg.png', b'bad')}, headers=SESSION)
    events = [json.loads(line) for line in response.text.splitlines()]
    assert len(events) == 1 and events[0]['type'] == 'error'


def test_fdi_failure_blocks_downstream_but_saves_report(client):
    calls = install_workflow_models()
    class BrokenFDI:
        def predict(self, rgb): raise RuntimeError('test failure')
    app.state.pipeline.models['fdi'] = BrokenFDI()
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION).json()
    assert calls == []
    assert [s['status'] for s in result['workflow']][2:] == ['failed', 'blocked', 'blocked', 'completed']
    assert client.get(result['final_report_url'], headers=SESSION).status_code == 200


def test_new_previews_are_deleted_at_expiry(client):
    from sqlalchemy.orm import Session
    from app.storage import Case
    install_workflow_models()
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION).json()
    storage = app.state.storage
    paths = [storage.local_path(result['case_id'], kind) for kind in ('impaction-preview', 'bone-loss-preview')]
    assert all(path.exists() for path in paths)
    with Session(storage.engine) as db:
        case = db.get(Case, result['case_id']); case.expires = time.time()-1; db.commit()
    storage.cleanup()
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize('missing', ['impaction', 'bone_loss'])
def test_missing_model_never_substitutes_findings(client, missing):
    install_workflow_models()
    del app.state.pipeline.models[missing]
    app.state.pipeline.status[missing] = 'unavailable: Model artifacts missing.'
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())},
                         data={'pixel_spacing_mm': '.1'}, headers=SESSION).json()
    assert next(s for s in result['workflow'] if s['id'] == missing)['status'] == 'blocked'
    if missing == 'impaction':
        assert result['impaction_preview_url'] is None
        for finding in result['third_molars']:
            assert all(finding[key] is None for key in ('impacted', 'impaction_confidence', 'angulation', 'angulation_confidence', 'gradcam_url'))
    else:
        assert result['bone_loss_preview_url'] is None
        for finding in result['bone_loss_results']:
            assert not finding['assessable']
            assert finding['mean_bone_loss_mm'] is None and finding['sites'] == []
            assert finding['severity'] == 'Not assessed'


def test_empty_fdi_does_not_invent_teeth_or_findings(client):
    calls = install_workflow_models()
    class EmptyFDI:
        def predict(self, rgb): return []
    app.state.pipeline.models['fdi'] = EmptyFDI()
    result = client.post('/module2/predict', files={'file': ('opg.png', payload())}, headers=SESSION).json()
    assert calls == []
    assert result['teeth'] == result['third_molars'] == result['bone_loss_results'] == []
    assert result['impaction_preview_url'] is None and result['bone_loss_preview_url'] is None
    report = client.get(result['final_report_url'], headers=SESSION).json()
    assert report['teeth'] == report['third_molars'] == report['bone_loss_results'] == []
