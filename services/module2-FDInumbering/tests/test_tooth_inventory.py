import io
import json
import unittest
import zipfile

from PIL import Image

from dental_fdi.analysis.dental_analysis import FDI_CODES, select_detections
from dental_fdi.analysis.tooth_inventory import FDI_ARCHES, apply_missing_review, tooth_inventory
from dental_fdi.reporting.prediction_exports import png_bytes, process_batch_item, review_batch_presence, build_batch_archive


class ToothInventoryTests(unittest.TestCase):
    def test_reference_chart_order(self):
        self.assertEqual(FDI_ARCHES['Upper right'], ('18','17','16','15','14','13','12','11'))
        self.assertEqual(FDI_ARCHES['Upper left'], ('21','22','23','24','25','26','27','28'))
        self.assertEqual(FDI_ARCHES['Lower right'], ('48','47','46','45','44','43','42','41'))
        self.assertEqual(FDI_ARCHES['Lower left'], ('31','32','33','34','35','36','37','38'))

    def test_missing_16_24_33_do_not_shift_other_teeth(self):
        missing = {'16', '24', '33'}
        rows = [{'pred_fdi': fdi, 'conf': .9, 'xyxy': [10, 10, 40, 60]} for fdi in FDI_CODES if fdi not in missing]
        boxes = select_detections(rows, 100, 100)
        self.assertEqual({b['fdi'] for b in boxes}, set(FDI_CODES) - missing)
        inventory = tooth_inventory(boxes, missing)
        self.assertEqual(len(inventory), 32)
        self.assertEqual([r['FDI'] for r in inventory if r['Status'] == 'Confirmed missing'], ['16','24','33'])
        self.assertEqual(sum(r['Status'] == 'Detected' for r in inventory), 29)
        self.assertTrue(all(r['Box'] is None for r in inventory if r['FDI'] in missing))

    def test_nondetection_is_not_confirmation_of_absence(self):
        inventory = tooth_inventory([])
        self.assertTrue(all(r['Status'] == 'Not detected — review' for r in inventory))
        boxes = [{'fdi': '17', 'xyxy': [10, 10, 40, 60], 'conf': .9}]
        kept, excluded = apply_missing_review(boxes, ['16'])
        self.assertEqual(kept, boxes)
        self.assertEqual(excluded, [])
        with self.assertRaises(ValueError):
            tooth_inventory(boxes, ['17'])

    def test_batch_review_removes_box_and_is_reversible_without_rerunning_model(self):
        image = Image.new('RGB', (200, 200), 'gray')
        result = process_batch_item('test.png', png_bytes(image), lambda _: [
            {'pred_fdi': '16', 'conf': .9, 'xyxy': [10, 10, 40, 60]},
            {'pred_fdi': '17', 'conf': .9, 'xyxy': [60, 10, 90, 60]}],
            render_options={'show_labels': False})
        reviewed = review_batch_presence(result, ['16'], {'show_labels': False})
        self.assertEqual([b['fdi'] for b in reviewed['detections']], ['17'])
        with Image.open(io.BytesIO(reviewed['overlay'])) as overlay:
            self.assertEqual(overlay.getpixel((10, 10)), (128, 128, 128))
        restored = review_batch_presence(reviewed, [], {'show_labels': False})
        self.assertEqual(len(restored['detections']), 2)
        self.assertEqual(restored['overlay'], result['overlay'])
        with zipfile.ZipFile(io.BytesIO(build_batch_archive([reviewed], {}))) as archive:
            report = json.loads(archive.read('predictions.json'))['images'][0]
            self.assertEqual(report['confirmed_missing'], ['16'])
            self.assertFalse(any(k.startswith('_') for k in report))
            self.assertIn('Confirmed missing', archive.read('tooth_status.csv').decode('utf-8-sig'))


if __name__ == '__main__':
    unittest.main()
