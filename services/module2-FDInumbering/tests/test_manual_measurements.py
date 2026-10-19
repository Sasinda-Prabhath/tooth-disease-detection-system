import io
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from dental_fdi.analysis.manual_geometry import line_angle, measure_bone_sites, measure_reference_lines
from dental_fdi.analysis.bone_pose import check_bone_dataset, context_crop, canonical_bone_crop, point_to_crop, point_from_crop
from dental_fdi.reporting.training_examples import export_fdi_example, export_bone_example
from dental_fdi.ui.box_drawing import put_manual_box
from dental_fdi.data.prepare_bone_crops import prepare


class ManualMeasurementTests(unittest.TestCase):
    def test_reference_line_angles_are_geometric_and_endpoint_invariant(self):
        line=[[20,20],[100,20]]
        red=[[20,20],[20+80*math.cos(math.radians(20)),20+80*math.sin(math.radians(20))]]
        self.assertAlmostEqual(line_angle(line,red),20)
        self.assertAlmostEqual(line_angle(line,red[::-1]),20)
        result=measure_reference_lines(line+[[20,20],[20,100]]+red,200,200)
        self.assertAlmostEqual(result['red_to_y_degrees'],70)
        self.assertIn('not a Winter',result['method'])

    def test_reviewed_bone_ratio_is_scale_invariant(self):
        points=[[20,20],[20,60],[20,220],[80,20],[80,80],[80,220]]
        first=measure_bone_sites(points,300,300)
        second=measure_bone_sites([[2*x,2*y] for x,y in points],600,600)
        self.assertAlmostEqual(first['sites']['mesial']['radiographic_bone_level_percent'],20)
        self.assertAlmostEqual(first['sites']['distal']['radiographic_bone_level_percent'],30)
        self.assertEqual(first['maximum_percent'],second['maximum_percent'])
        self.assertIsNone(first['millimetres'])
        self.assertIn('Not determined',first['clinical_stage'])

    def test_invalid_or_missing_bone_landmarks_do_not_generate_measurements(self):
        for points in [[],[[20,20]]*6,[[20,50],[20,10],[20,150]]*2]:
            with self.assertRaises(ValueError):
                measure_bone_sites(points,200,200)

    def test_drawing_one_tooth_does_not_renumber_neighbors(self):
        boxes=[{'fdi':'15','pred_fdi':'15','xyxy':[10,10,30,80],'conf':.9},
               {'fdi':'17','pred_fdi':'17','xyxy':[80,10,110,80],'conf':.9}]
        updated=put_manual_box(boxes,'16',[40,10,70,80],200,100)
        self.assertEqual([b['fdi'] for b in updated],['15','16','17'])
        self.assertEqual(updated[0],boxes[0]);self.assertEqual(updated[2],boxes[1])
        self.assertEqual(updated[1]['source'],'Manual correction')

    def test_fdi_training_export_contains_clean_pixels_and_separate_labels(self):
        image=Image.new('RGB',(200,300),'gray')
        boxes=put_manual_box([],'17',[10,20,50,250],200,300)
        with zipfile.ZipFile(io.BytesIO(export_fdi_example(image,boxes,['16']))) as archive:
            names=archive.namelist()
            pixels=next(name for name in names if name.startswith('images/'))
            labels=next(name for name in names if name.startswith('labels/'))
            presence=next(name for name in names if name.startswith('presence/'))
            with Image.open(io.BytesIO(archive.read(pixels))) as saved:
                self.assertEqual(saved.tobytes(),image.tobytes())
            self.assertEqual(archive.read(labels).decode().split()[0],'6')
            self.assertEqual(json.loads(archive.read(presence))['status']['16'],'missing')

    def test_bone_export_has_six_landmarks_and_roundtrips_dataset_validation(self):
        image=Image.new('RGB',(200,300),'gray')
        box={'fdi':'17','xyxy':[10,10,100,250]}
        points=[[20,20],[20,60],[20,220],[80,20],[80,80],[80,220]]
        with zipfile.ZipFile(io.BytesIO(export_bone_example(image,box,points,'abc123'))) as archive:
            label=archive.read('labels/opg_abc123_FDI17.txt').decode()
            self.assertEqual(len(label.split()),23)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                for split,color in [('train','black'),('val','white')]:
                    (root/'images'/split).mkdir(parents=True)
                    (root/'labels'/split).mkdir(parents=True)
                    Image.new('RGB',(200,300),color).save(root/'images'/split/'tooth.png')
                    (root/'labels'/split/'tooth.txt').write_text(label)
                self.assertEqual(check_bone_dataset(root),[])
                (root/'labels/train/tooth.txt').write_text('0 .5 .5 1 1\n')
                self.assertTrue(any('23 values' in e for e in check_bone_dataset(root)))

    def test_crop_preparation_preserves_splits_and_never_invents_keypoints(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';output=Path(directory)/'bone'
            for split,color in [('train','black'),('val','white')]:
                (source/'images'/split).mkdir(parents=True)
                (source/'labels'/split).mkdir(parents=True)
                Image.new('RGB',(200,300),color).save(source/'images'/split/'patient.png')
                (source/'labels'/split/'patient.txt').write_text('5 .5 .5 .2 .6\n')
            records=prepare(source,output)
            self.assertEqual({r['split'] for r in records},{'train','val'})
            self.assertTrue(all(r['fdi']=='16' for r in records))
            self.assertEqual(list((output/'labels').rglob('*.txt')),[])
            self.assertEqual(len(list((output/'images').rglob('*.png'))),2)
            self.assertTrue(check_bone_dataset(output))
            with self.assertRaises(ValueError):
                prepare(source,output)

    def test_bone_crop_mirroring_maps_back_to_the_correct_patient_side(self):
        image=Image.new('RGB',(200,300),'gray')
        image.putpixel((20,30),(255,0,0))
        for fdi,expected in [('16',True),('26',False),('36',False),('46',True)]:
            crop,bounds,mirrored=canonical_bone_crop(image,[10,10,100,250],fdi)
            local=point_to_crop([20,30],bounds,mirrored)
            self.assertEqual(mirrored,expected)
            self.assertEqual(point_from_crop(local,bounds,mirrored),[20,30])
            self.assertEqual(crop.getpixel(tuple(map(int,local))),(255,0,0))


if __name__=='__main__':
    unittest.main()
