import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from dental_fdi.analysis.periodontal_measurement import measure_periodontal_sites
from dental_fdi.training.train_periodontal import check_patient_manifest


class PeriodontalProtocolTests(unittest.TestCase):
    def test_axial_measurement_does_not_count_lateral_offset_as_depth(self):
        points=[[20,20],[50,60],[20,220],[80,20],[110,80],[80,220]]
        result=measure_periodontal_sites(points,300,300)
        self.assertAlmostEqual(result['sites']['mesial']['radiographic_bone_level_percent'],20)
        self.assertAlmostEqual(result['sites']['distal']['radiographic_bone_level_percent'],30)
        self.assertEqual(result['sites']['mesial']['crest_projection'],[20,60])
        self.assertIsNone(result['millimetres'])

    def test_rotation_and_scaling_preserve_ratios(self):
        points=[[20,20],[20,60],[20,220],[80,20],[80,80],[80,220]]
        base=measure_periodontal_sites(points,300,300)
        rotated=[[600-2*y,2*x] for x,y in points]
        other=measure_periodontal_sites(rotated,600,600)
        self.assertAlmostEqual(base['maximum_percent'],other['maximum_percent'])

    def test_wrong_side_and_zero_axis_rejected(self):
        for points in [[[20,20],[20,10],[20,220]]*2,[[20,20]]*6]:
            with self.assertRaises(ValueError):measure_periodontal_sites(points,300,300)

    def test_patient_leakage_and_missing_manifest_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for split in ('train','val','test'):
                (root/'images'/split).mkdir(parents=True)
                Image.new('RGB',(10,10)).save(root/'images'/split/(split+'.png'))
            manifest=root/'patients.csv'
            manifest.write_text('image,patient_id,split\ntrain.png,P1,train\nval.png,P1,val\n')
            errors=check_patient_manifest(root,manifest)
            self.assertTrue(any('multiple splits' in e for e in errors))
            self.assertTrue(any('No patient ID' in e for e in errors))
            manifest.write_text('image,patient_id,split\ntrain.png,P1,train\nval.png,P2,val\ntest.png,P3,test\n')
            self.assertEqual(check_patient_manifest(root,manifest),[])
