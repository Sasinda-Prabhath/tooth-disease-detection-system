import unittest
from unittest.mock import patch
from PIL import Image
from dental_fdi.ui.reference_lines import render_manual_measurements
from dental_fdi.reporting.prediction_exports import render_detection_overlay


class ManualIsolationTests(unittest.TestCase):
    def test_empty_manual_drawings_never_generate_automatic_lines(self):
        image=Image.new('RGB',(300,300),'gray')
        boxes=[{'fdi':'38','xyxy':[100,100,200,250]}]
        with patch('dental_fdi.analysis.image_angulation.estimate_image_angles',side_effect=AssertionError('Automatic estimator must not run')):
            overlay,rows=render_manual_measurements(image,boxes,{'38':[]})
        self.assertEqual(overlay.crop((0,0,*image.size)).tobytes(),render_detection_overlay(image,boxes,show_conf=False).tobytes())
        self.assertIsNone(rows[0]['selected_angle_degrees'])

    def test_user_lines_supply_manual_result_without_classification_guess(self):
        points=[[10,150],[290,150],[150,10],[150,290],[50,50],[250,250]]
        _,rows=render_manual_measurements(Image.new('RGB',(300,300)),[],{'38':points})
        self.assertAlmostEqual(rows[0]['selected_angle_degrees'],45)
        self.assertEqual(rows[0]['source'],'Manual drawing only')
        self.assertIn('Unavailable',rows[0]['classification'])
        self.assertEqual(rows[0]['points'],points)

    def test_manual_second_axis_supplies_classification_without_estimation(self):
        points=[[10,150],[290,150],[150,10],[150,290],[100,200],[150,150],[200,200],[200,100]]
        with patch('dental_fdi.analysis.image_angulation.estimate_image_angles',side_effect=AssertionError('No automatic axes')):
            _,rows=render_manual_measurements(Image.new('RGB',(300,300)),[],{'48':points})
        self.assertAlmostEqual(rows[0]['angle_to_second_molar_degrees'],45)
        self.assertEqual(rows[0]['classification'],'Mesioangular')

    def test_manual_shading_is_yellow(self):
        points=[[10,150],[290,150],[150,10],[150,290],[50,50],[250,250]]
        overlay,_=render_manual_measurements(Image.new('RGB',(300,300)),[],{'38':points})
        self.assertEqual(overlay.getpixel((139,146)),(255,219,53))

    def test_requested_signed_horizontal_categories_ignore_click_order(self):
        from dental_fdi.ui.reference_lines import manual_summary
        import math
        guides=[[0,150],[299,150],[150,0],[150,299]]
        for degrees,category in [(-49.95,'Distoangular'),(45.50,'Mesioangular')]:
            dx=80*math.cos(math.radians(abs(degrees)))*(1 if degrees>0 else -1)
            axis=[[150,220],[150+dx,220-80*math.sin(math.radians(abs(degrees)))]]
            for ordered in (axis,axis[::-1]):
                r=manual_summary('38',guides+ordered,300,300)
                self.assertAlmostEqual(r['signed_horizontal_degrees'],degrees)
                self.assertEqual(r['screen_category'],category)
                self.assertIsNone(r['angle_to_second_molar_degrees'])
