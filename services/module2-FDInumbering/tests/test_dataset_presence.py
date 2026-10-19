import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.data.dataset_checks import check_dataset


class DatasetPresenceTests(unittest.TestCase):
    def test_missing_teeth_cannot_have_training_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, color in [('train', 'black'), ('val', 'white')]:
                for subdir in ['images', 'labels', 'presence']:
                    (root / subdir / split).mkdir(parents=True)
                Image.new('RGB', (100, 100), color).save(root / 'images' / split / 'opg.png')
                # 5 is FDI 16, not FDI 15 and not the label number 16.
                (root / 'labels' / split / 'opg.txt').write_text('5 .5 .5 .2 .2\n')
            statuses = dict.fromkeys(FDI_CODES, 'uncertain')
            statuses['16'] = 'missing'
            presence = root / 'presence/train/opg.json'
            presence.write_text(json.dumps({'schema': 'fdi_presence_annotation_v1', 'status': statuses}))
            self.assertTrue(any('16: marked missing' in e for e in check_dataset(root)))
            statuses['16'] = 'present'
            presence.write_text(json.dumps({'schema': 'fdi_presence_annotation_v1', 'status': statuses}))
            self.assertEqual(check_dataset(root), [])


if __name__ == '__main__':
    unittest.main()
