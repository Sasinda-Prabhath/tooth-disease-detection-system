"""Verify per-image missing-tooth review and actual ZIP contents in a browser."""
import argparse
import json
import zipfile
from playwright.sync_api import sync_playwright, expect
from dental_fdi.paths import DATASET_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8527')
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(accept_downloads=True)
        page.goto(args.url)
        page.get_by_role('tab', name='Bulk predictions').click(timeout=90000)
        paths = sorted((DATASET_DIR / 'images/val').glob('*.png'))[:2]
        page.locator('input[type=file][multiple]').set_input_files([str(path) for path in paths])
        page.get_by_role('button', name='Run batch predictions', exact=True).click()
        page.get_by_text('ZIP download canceled or not starting?', exact=True).wait_for(timeout=120000)

        def read_zip():
            page.keyboard.press('Escape')
            link = page.get_by_role('link', name='Download ZIP directly', exact=True)
            expander = page.get_by_test_id('stExpander').filter(has_text='ZIP download canceled or not starting?').locator('details')
            if expander.get_attribute('open') is None:
                expander.locator('summary').click()
            link.wait_for(state='visible')
            with page.expect_download(timeout=30000) as pending:
                page.get_by_role('link', name='Download ZIP directly', exact=True).click()
            download = pending.value
            assert download.failure() is None, download.failure()
            with zipfile.ZipFile(download.path()) as archive:
                assert archive.testzip() is None
                assert 'tooth_status.csv' in archive.namelist()
                return json.loads(archive.read('predictions.json'))

        initial = read_zip()
        chosen = []
        for fdi in ['16', '24', '33']:
            widget = page.get_by_test_id('stMultiSelect').filter(has_text='Confirmed missing teeth for this image (reviewed)')
            field = widget.locator('input')
            field.click()
            field.fill(fdi)
            page.get_by_role('option', name=fdi, exact=True).click()
            expect(widget).to_contain_text(fdi)
            chosen.append(fdi)
            page.get_by_text('Reviewed missing for this image: ' + ', '.join(chosen), exact=True).wait_for(timeout=30000)
            page.wait_for_timeout(400)
        # Wait for the final rerun to update the report before reading its link.
        expect(page.get_by_text('Running...', exact=True)).to_have_count(0, timeout=30000)
        reviewed = read_zip()
        image = reviewed['images'][0]
        assert set(image['confirmed_missing']) == {'16', '24', '33'}, image['confirmed_missing']
        assert len(image['tooth_status']) == 32
        assert {t['FDI'] for t in image['tooth_status'] if t['Status'] == 'Confirmed missing'} == {'16', '24', '33'}
        assert image['detections'] == [b for b in initial['images'][0]['detections'] if b['fdi'] not in {'16', '24', '33'}]
        assert reviewed['images'][1]['confirmed_missing'] == []
        assert not page.locator('[data-testid=stException]').count()
        print('PASS browser: 16/24/33 missing review; neighboring codes/boxes unchanged; other image unaffected; downloaded ZIP inventories valid')
        browser.close()


if __name__ == '__main__':
    main()
