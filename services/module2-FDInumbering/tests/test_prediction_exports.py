import io
import json
import unittest
import zipfile

from PIL import Image

from dental_fdi.reporting.prediction_exports import build_batch_archive, png_bytes, process_batch_item, render_detection_overlay


class BatchExportTests(unittest.TestCase):
    def test_predictions_and_overlay_share_oriented_pixels(self):
        image = Image.new('RGB', (80, 50), 'black')
        image.putpixel((0, 0), (255, 0, 0))
        seen = []

        def predict(data):
            with Image.open(io.BytesIO(data)) as decoded:
                seen.append(decoded.getpixel((79, 0)))
            return [{'pred_fdi': '18', 'conf': .9, 'xyxy': [-5, 4, 30, 40]},
                    {'pred_fdi': '18', 'conf': .5, 'xyxy': [1, 4, 30, 40]},
                    {'pred_fdi': '28', 'conf': .1, 'xyxy': [40, 4, 60, 40]}]

        result = process_batch_item('opg.jpg', png_bytes(image), predict, mirror=True,
            render_options={'show_labels': False})
        self.assertEqual(seen, [(255, 0, 0)])
        self.assertEqual(len(result['detections']), 1)
        self.assertEqual(result['detections'][0]['xyxy'], [0, 4, 30, 40])
        with Image.open(io.BytesIO(result['overlay'])) as overlay:
            self.assertEqual(overlay.size, image.size)
            self.assertEqual(overlay.getpixel((79, 0)), (255, 0, 0))
            self.assertEqual(overlay.getpixel((0, 4)), (0, 255, 127))

    def test_duplicate_names_and_failed_images_export_without_overwriting(self):
        source = png_bytes(Image.new('RGB', (80, 50), 'gray'))
        results = [process_batch_item('../../scan.jpg', source, lambda data: [], index=i) for i in (1, 2)]
        results.append({'input_name': 'bad.jpg', 'status': 'Failed', 'error': 'Invalid image', 'detections': []})
        archive = build_batch_archive(results, {'confidence': .35})
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            names = bundle.namelist()
            overlays = [name for name in names if name.startswith('overlays/')]
            self.assertEqual(len(set(overlays)), 2)
            self.assertFalse(any('..' in name for name in names))
            report = json.loads(bundle.read('predictions.json'))
            self.assertEqual(report['images'][2]['status'], 'Failed')
            self.assertEqual(report['images'][0]['status'], 'No detections')
            self.assertIn('Invalid image', bundle.read('summary.csv').decode('utf-8-sig'))
            self.assertIn('Confidence', bundle.read('predictions.csv').decode('utf-8-sig'))

    def test_exif_orientation_is_applied_before_prediction(self):
        image = Image.new('RGB', (80, 50), 'gray')
        exif = image.getexif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', exif=exif)
        sizes = []

        def predict(data):
            with Image.open(io.BytesIO(data)) as decoded:
                sizes.append(decoded.size)
            return []

        result = process_batch_item('rotated.jpg', buffer.getvalue(), predict)
        self.assertEqual(sizes, [(50, 80)])
        self.assertEqual((result['width'], result['height']), (50, 80))

    def test_display_toggles_export_original_pixels(self):
        image = Image.new('RGB', (80, 50), 'gray')
        overlay = render_detection_overlay(image, [{'fdi': '18', 'conf': .9, 'xyxy': [0, 0, 79, 49]}],
                                           show_boxes=False, show_labels=False)
        self.assertEqual(overlay.tobytes(), image.tobytes())


if __name__ == '__main__':
    unittest.main()
