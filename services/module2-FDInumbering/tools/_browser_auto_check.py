"""Browser smoke test for the rebuilt bulk and three-model workflow."""
import argparse
import json
import zipfile
from pathlib import Path
from playwright.sync_api import sync_playwright
from dental_fdi.paths import DATASET_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8525')
    args = parser.parse_args()
    paths = sorted((DATASET_DIR / 'images/val').glob('*.png'))[:2]
    if len(paths) < 2:
        raise ValueError('Two validation PNGs are required for this local smoke test')
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(accept_downloads=True)
        page.goto(args.url)
        page.get_by_role('tab', name='Bulk predictions').click(timeout=90000)
        uploads = [{'name': path.name, 'mimeType': 'image/png', 'buffer': path.read_bytes()} for path in paths]
        uploads.append({'name': 'invalid.png', 'mimeType': 'image/png', 'buffer': b'invalid image'})
        page.locator('input[type=file][multiple]').set_input_files(uploads)
        page.get_by_role('button', name='Run batch predictions', exact=True).click()
        page.get_by_text('ZIP download canceled or not starting?', exact=True).click(timeout=120000)
        for attempt in range(2):
            with page.expect_download(timeout=30000) as pending:
                page.get_by_role('link', name='Download ZIP directly', exact=True).click()
            download = pending.value
            assert download.failure() is None, download.failure()
            with zipfile.ZipFile(download.path()) as archive:
                assert archive.testzip() is None
                assert len([name for name in archive.namelist() if name.endswith('.png')]) == 2
                report = json.loads(archive.read('predictions.json'))
                assert len(report['images']) == 3
                assert sum(r['status'] == 'Failed' for r in report['images']) == 1
                boxes = [box for r in report['images'] for box in r.get('detections', [])]
                assert boxes
                assert all(set(b['model_contributions']) == {'yolo26','efficientnet_b0','resnet50'} for b in boxes)
                print(f'Download {attempt+1}: verified ZIP, two overlays, failure report and {len(boxes)} three-model predictions')
        assert not page.locator('[data-testid=stException]').count()
        browser.close()


if __name__ == '__main__':
    main()
