"""Private objects, expiring case capabilities, and metadata without raw URLs."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import time
from PIL import Image
from sqlalchemy import create_engine, Column, String, Float, Text, select
from sqlalchemy.orm import DeclarativeBase, Session

ROOT = Path(__file__).resolve().parents[1]

class Base(DeclarativeBase):
    pass

class Case(Base):
    __tablename__ = 'module2_cases'
    id = Column(String, primary_key=True)
    session_hash = Column(String, nullable=False)
    expires = Column(Float, nullable=False)
    metadata_json = Column(Text, nullable=False)

class Storage:
    def __init__(self):
        self.root = Path(os.getenv('MODULE2_STORAGE_DIR', str(ROOT/'outputs'))).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_root = self.root/'raw_protected'
        self.raw_root.mkdir(exist_ok=True)
        self.engine = create_engine(os.getenv('DATABASE_URL', f'sqlite:///{self.root / "cases.sqlite3"}'))
        Base.metadata.create_all(self.engine)
        self.bucket = os.getenv('S3_BUCKET')
        self.s3 = None
        if self.bucket:
            import boto3
            self.s3 = boto3.client('s3', endpoint_url=os.getenv('S3_ENDPOINT_URL'))
        self.ttl = int(os.getenv('CASE_TTL_SECONDS', '3600'))

    def create(self, session, metadata):
        case_id = 'case_'+datetime.now(timezone.utc).strftime('%Y%m%d')+'_'+secrets.token_hex(8)
        with Session(self.engine) as db:
            db.add(Case(id=case_id, session_hash=sha256(session.encode()).hexdigest(), expires=time.time()+self.ttl, metadata_json=json.dumps(metadata)))
            db.commit()
        return case_id

    def authorised(self, case_id, session):
        with Session(self.engine) as db:
            case = db.get(Case, case_id)
            return bool(case and case.expires > time.time() and secrets.compare_digest(case.session_hash, sha256(session.encode()).hexdigest()))

    def put(self, case_id, kind, data):
        # Raw bytes never enter S3 or metadata and never use uploaded filenames as paths.
        if kind == 'raw':
            (self.raw_root/f'{case_id}.png').write_bytes(data)
        elif self.s3:
            self.s3.put_object(Bucket=self.bucket, Key=f'{case_id}/{kind}', Body=data)
        else:
            path = self.local_path(case_id, kind)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)

    def local_path(self, case_id, kind):
        if kind == 'processed':
            return self.root/'processed'/f'{case_id}_processed.png'
        if kind == 'fdi-preview':
            return self.root/'fdi_preview'/f'{case_id}_fdi.png'
        suffix = '.json' if kind == 'final-report' else '.png'
        return self.root/'final_reports'/f'{case_id}_{kind}{suffix}'

    def image(self, case_id, kind, rgb):
        data = BytesIO()
        Image.fromarray(rgb).save(data, format='PNG')
        self.put(case_id, kind, data.getvalue())

    def get(self, case_id, kind):
        if kind == 'raw':
            return (self.raw_root/f'{case_id}.png').read_bytes()
        if self.s3:
            return self.s3.get_object(Bucket=self.bucket, Key=f'{case_id}/{kind}')['Body'].read()
        return self.local_path(case_id, kind).read_bytes()

    def cleanup(self):
        # IDs come only from the database; avoid recursive removal of computed paths.
        with Session(self.engine) as db:
            expired = db.scalars(select(Case).where(Case.expires <= time.time())).all()
            for case in expired:
                (self.raw_root/f'{case.id}.png').unlink(missing_ok=True)
                for kind in ('processed', 'fdi-preview', 'impaction-preview', 'bone-loss-preview', 'final-image', 'final-report', 'gradcam-18', 'gradcam-28', 'gradcam-38', 'gradcam-48'):
                    self.local_path(case.id, kind).unlink(missing_ok=True)
                if self.s3:
                    for page in self.s3.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=f'{case.id}/'):
                        for obj in page.get('Contents', []):
                            self.s3.delete_object(Bucket=self.bucket, Key=obj['Key'])
                db.delete(case)
            db.commit()
