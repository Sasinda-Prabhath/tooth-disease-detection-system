import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from dental_fdi.analysis.image_angulation import estimate_tooth_axis, estimate_image_angles
from dental_fdi.analysis.pose_pipeline import check_pose_dataset, automatic_angles

class AutomaticAxisTests(unittest.TestCase):
    def synthetic(self,angle):
        pixels=np.full((200,200,3),20,dtype=np.uint8)
        cv2.ellipse(pixels,(100,100),(22,70),angle,0,360,(220,220,220),-1)
        return Image.fromarray(pixels)

    def test_pixel_content_changes_axis_inside_identical_box(self):
        box={'fdi':'48','xyxy':[0,0,200,200]}
        a,error=estimate_tooth_axis(self.synthetic(0),box)
        b,error=estimate_tooth_axis(self.synthetic(35),box)
        self.assertIsNotNone(a);self.assertIsNotNone(b)
        va=np.subtract(a['points'][1],a['points'][0]);vb=np.subtract(b['points'][1],b['points'][0])
        theta=np.degrees(np.arccos(abs(np.dot(va,vb)/np.linalg.norm(va)/np.linalg.norm(vb))))
        self.assertAlmostEqual(theta,35,delta=2)

    def test_moment_estimator_follows_long_axis_not_perpendicular(self):
        from unittest.mock import patch
        for angle in (0,35,-35,90):
            with self.subTest(angle=angle), patch('dental_fdi.analysis.image_angulation._component_axis',return_value=None):
                axis,reason=estimate_tooth_axis(self.synthetic(angle),{'fdi':'48','xyxy':[0,0,200,200]})
            self.assertIsNotNone(axis,reason)
            vector=np.subtract(axis['points'][1],axis['points'][0])
            expected=np.array([-np.sin(np.radians(angle)),np.cos(np.radians(angle))])
            error=np.degrees(np.arccos(np.clip(abs(np.dot(vector,expected))/np.linalg.norm(vector),0,1)))
            self.assertLess(error,2)

    def test_flat_crop_is_rejected_not_assigned_zero_degrees(self):
        axis,reason=estimate_tooth_axis(Image.new('RGB',(100,100),'gray'),{'fdi':'48','xyxy':[0,0,100,100]})
        self.assertIsNone(axis);self.assertIn('contrast',reason)

    def test_missing_reference_does_not_produce_angle(self):
        rows,axes=estimate_image_angles(self.synthetic(25),[{'fdi':'48','xyxy':[0,0,200,200]}])
        self.assertIsNone(rows[-1]['Estimated angle (degrees)'])
        self.assertIn('47: not detected',rows[-1]['Status'])

    def test_pose_pairing_and_signed_output(self):
        axes=[{'fdi':'48','points':[[100,100],[150,50]],'keypoint_confidence':[.9,.9]},
              {'fdi':'47','points':[[200,100],[200,50]],'keypoint_confidence':[.9,.9]}]
        rows=automatic_angles(axes,300,300)
        self.assertEqual(rows[-1]['Angle (degrees)'],45)
        self.assertEqual(rows[-1]['Classification'],'Mesioangular')
        self.assertIsNone(rows[0]['Angle (degrees)'])

    def test_box_only_labels_cannot_be_pose_training_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for split,value in [('train',100),('val',200)]:
                (root/'images'/split).mkdir(parents=True)
                (root/'labels'/split).mkdir(parents=True)
                Image.new('RGB',(20,20),(value,0,0)).save(root/'images'/split/'test.png')
                (root/'labels'/split/'test.txt').write_text('0 .5 .5 .5 .5\n')
            errors=check_pose_dataset(root)
            self.assertEqual(len(errors),2)
            self.assertTrue(all('root_x' in e for e in errors))

class AutomaticAvailabilityTests(unittest.TestCase):
    def test_failed_reference_axis_does_not_use_box_vertical(self):
        from unittest.mock import patch
        image=Image.new('RGB',(200,200))
        boxes=[{'fdi':'48','xyxy':[0,0,90,190]}, {'fdi':'47','xyxy':[100,0,190,190]}]
        with patch('dental_fdi.analysis.image_angulation.estimate_tooth_axis',
                   side_effect=[({'points':[[20,170],[70,70]]},None),(None,'Insufficient contrast')]):
            rows,_=estimate_image_angles(image,boxes)
        self.assertIsNone(rows[-1]['Angle (degrees)'])
        self.assertIn('Insufficient contrast',rows[-1]['Status'])

if __name__=='__main__':
    unittest.main()
