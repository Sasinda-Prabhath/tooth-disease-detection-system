from hashlib import sha256
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from app.schemas import Prediction, ImpactionResult, BoneResult, WorkflowStage
from app.services.preprocessing import decode, preprocess
from app.services.privacy_masking import PrivacyMasker
from app.services.fdi_detection import FDIDetector
from app.services.impaction import ImpactionClassifier, crop_and_geometry
from app.services.bone_loss import BoneSegmenter
from app.services.report import annotate
from app.storage import ROOT

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, storage):
        self.storage = storage
        self.lock = threading.Lock()  # Ultralytics/Paddle predictors and Grad-CAM are stateful.
        self.models, self.status, self.versions = {}, {}, {}
        root = Path(os.getenv('MODELS_DIR', str(ROOT/'models')))
        specs = {'privacy': (PrivacyMasker, root/'privacy'), 'fdi': (FDIDetector, root/'fdi_yolo'/'best.pt'),
                 'impaction': (ImpactionClassifier, root/'impaction'/'best.pt'), 'bone_loss': (BoneSegmenter, root/'bone_loss')}
        for name, (factory, path) in specs.items():
            try:
                if not path.exists():
                    raise FileNotFoundError('Model artifacts missing.')
                self.models[name] = factory(path)
                digest = sha256()
                for artifact in sorted(path.rglob('*')) if path.is_dir() else [path]:
                    if artifact.is_file():
                        digest.update(artifact.name.encode())
                        with artifact.open('rb') as file:
                            for chunk in iter(lambda: file.read(1024*1024), b''):
                                digest.update(chunk)
                self.versions[name] = digest.hexdigest()
                self.status[name] = 'loaded'
            except Exception as exc:
                self.status[name] = f'unavailable: {type(exc).__name__}'
                logger.warning('Model %s unavailable (%s)', name, type(exc).__name__)

    def health(self):
        return {'status': 'ok' if all(s == 'loaded' for s in self.status.values()) else 'degraded',
                'models': self.status.copy(), 'model_versions': self.versions.copy(), 'inference_mode': 'research_prototype'}

    def run(self, payload, filename, spacing, session, on_stage=None):
        with self.lock:
            return self._run(payload, filename, spacing, session, on_stage)

    def _run(self, payload, filename, spacing, session, on_stage=None):
        try:
            rgb, spacing, calibration = decode(payload, filename, spacing)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('Unable to decode this OPG; verify file format and installed DICOM pixel codecs.') from exc
        case_id = self.storage.create(session, {'uploaded_filename': Path(filename).name,
            'pixel_spacing_mm': spacing, 'model_versions': self.versions, 'uploaded_at': datetime.now(timezone.utc).isoformat()})
        base = f'/module2/cases/{case_id}'
        self.storage.image(case_id, 'raw', rgb)
        result = Prediction(case_id=case_id, model_status='ready' if self.health()['status'] == 'ok' else 'degraded',
            raw_image_url=base+'/raw', image_width=rgb.shape[1], image_height=rgb.shape[0],
            pixel_spacing_mm=spacing, calibration_source=calibration, model_versions=self.versions.copy(),
            workflow=[WorkflowStage(id=key, label=label) for key, label in (
                ('upload', 'Upload radiograph'), ('preprocessing', 'Prepare radiograph'),
                ('fdi', 'FDI tooth numbering'), ('impaction', 'Third-molar impaction and angulation'),
                ('bone_loss', 'Alveolar bone loss'), ('report', 'Final report'))])
        result.warnings += [f'{name}: {status}' for name, status in self.status.items() if status != 'loaded']

        def stage(key, status, detail=None, image=None):
            item = next(item for item in result.workflow if item.id == key)
            item.status, item.detail, item.image_url = status, detail, image
            if status in ('failed', 'blocked', 'partial'):
                result.model_status = 'degraded'
            if on_stage:
                on_stage(result.model_dump())

        def finish():
            # Even blocked cases get a report explaining which stages did not run.
            for item in result.workflow:
                if item.id != 'report' and item.status == 'pending':
                    stage(item.id, 'blocked', 'Required upstream stage did not complete.')
            stage('report', 'running')
            result.final_report_url = base+'/final-report'
            item = result.workflow[-1]
            item.status = 'completed'
            item.detail = 'Report includes findings, unavailable assessments and stage outcomes.'
            item.image_url = result.final_image_url
            self.storage.put(case_id, 'final-report', json.dumps(
                result.model_dump(exclude={'raw_image_url': True, 'workflow': {0: {'image_url'}}}), allow_nan=False).encode())
            if on_stage:
                on_stage(result.model_dump())
            return result

        stage('upload', 'completed', image=result.raw_image_url)
        stage('preprocessing', 'running')
        if 'privacy' not in self.models:
            result.warnings.append('Privacy masking unavailable. Processing and clinical predictions were withheld.')
            stage('preprocessing', 'blocked', result.warnings[-1])
            return finish()
        try:
            processed, warnings = preprocess(rgb, self.models['privacy'])
        except ValueError as exc:
            result.warnings.append(str(exc))
            stage('preprocessing', 'blocked', str(exc))
            return finish()
        except Exception:
            self.status['privacy'] = 'unavailable: inference failed'
            result.warnings.append('Privacy inference failed. Processing and predictions were withheld.')
            stage('preprocessing', 'failed', result.warnings[-1])
            return finish()
        result.warnings += warnings
        self.storage.image(case_id, 'processed', processed)
        result.processed_image_url = base+'/processed'
        stage('preprocessing', 'completed', image=result.processed_image_url)

        if 'fdi' not in self.models:
            stage('fdi', 'blocked', 'FDI model unavailable; dependent predictions withheld.')
            return finish()
        stage('fdi', 'running')
        try:
            result.teeth = self.models['fdi'].predict(processed)
        except Exception:
            self.status['fdi'] = 'unavailable: inference failed'
            result.warnings.append('FDI inference failed; dependent predictions withheld.')
            stage('fdi', 'failed', result.warnings[-1])
            return finish()
        self.storage.image(case_id, 'fdi-preview', annotate(processed, result.teeth))
        result.fdi_preview_url = base+'/fdi-preview'
        stage('fdi', 'partial' if any(t.status == 'uncertain' for t in result.teeth) else 'completed',
              f'{len(result.teeth)} teeth numbered.', result.fdi_preview_url)
        if not result.teeth:
            result.warnings.append('No teeth detected; this does not establish absence of disease.')

        molars = [t for t in result.teeth if t.fdi in ('18', '28', '38', '48')]
        # Coordinates and clean pixels flow between models; overlays are display artifacts.
        crops = {}
        for tooth in molars:
            if tooth.status == 'detected':
                try:
                    crops[tooth.fdi] = crop_and_geometry(processed, tooth, result.teeth)
                except Exception:
                    result.warnings.append(f'FDI {tooth.fdi}: unable to prepare a third-molar crop.')
        stage('impaction', 'running')
        impaction_failed = False
        for tooth in molars:
            prediction = ImpactionResult(fdi=tooth.fdi, reason='Classifier unavailable or FDI uncertain; crop may be unavailable.')
            if tooth.fdi in crops and 'impaction' in self.models:
                crop, _, features = crops[tooth.fdi]
                try:
                    prediction = self.models['impaction'].predict(crop, features, tooth.fdi)
                except Exception:
                    impaction_failed = True
                    self.status['impaction'] = 'unavailable: inference failed'
                    prediction.reason = 'Impaction classifier inference failed.'
                if prediction.impacted is not None:
                    try:
                        kind = f'gradcam-{tooth.fdi}'
                        self.storage.image(case_id, kind, self.models['impaction'].gradcam(crop, features))
                        prediction.gradcam_url = base+'/'+kind
                    except Exception:
                        # An explanation failure must not discard a successful prediction.
                        result.warnings.append(f'FDI {tooth.fdi}: Grad-CAM explanation unavailable.')
            result.third_molars.append(prediction)
        if not crops or 'impaction' not in self.models:
            stage('impaction', 'blocked', 'No reliable third-molar crops or impaction model unavailable.')
        else:
            self.storage.image(case_id, 'impaction-preview', annotate(processed, result.teeth, impactions=result.third_molars))
            result.impaction_preview_url = base+'/impaction-preview'
            partial = impaction_failed or any(t.impacted is None or (t.impacted and t.angulation is None) for t in result.third_molars)
            stage('impaction', 'partial' if partial else 'completed',
                  'Third-molar impaction and angulation assessed; see per-tooth results.', result.impaction_preview_url)

        # Finish ALL impaction assessments before starting any bone-loss assessment.
        # Bone segmentation depends on FDI crops, not the classifier outcome.
        stage('bone_loss', 'running')
        for tooth in molars:
            bone = BoneResult(fdi=tooth.fdi, reason='Segmentation unavailable or FDI uncertain; crop may be unavailable.')
            if tooth.fdi in crops and 'bone_loss' in self.models:
                crop, origin, _ = crops[tooth.fdi]
                try:
                    bone = self.models['bone_loss'].predict(crop, tooth, origin, spacing)
                except Exception:
                    self.status['bone_loss'] = 'unavailable: inference failed'
                    bone.reason = 'Bone segmentation inference failed.'
            result.bone_loss_results.append(bone)
        if not crops or 'bone_loss' not in self.models:
            stage('bone_loss', 'blocked', 'No reliable third-molar crops or bone-loss model unavailable.')
        else:
            self.storage.image(case_id, 'bone-loss-preview', annotate(processed, result.teeth, result.bone_loss_results, result.third_molars))
            result.bone_loss_preview_url = base+'/bone-loss-preview'
            stage('bone_loss', 'completed' if all(b.assessable for b in result.bone_loss_results) else 'partial',
                  'Third-molar sites only. Millimetre measurements require verified calibration and assessable landmarks.',
                  result.bone_loss_preview_url)
        result.warnings.append('CEJ/crest site extraction is a research heuristic requiring dentist review; severity is not assessed.')
        self.storage.image(case_id, 'final-image', annotate(processed, result.teeth, result.bone_loss_results, result.third_molars))
        result.final_image_url = base+'/final-image'
        return finish()
