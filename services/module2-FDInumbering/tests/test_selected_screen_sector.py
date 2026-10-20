import unittest
from PIL import Image, ImageDraw
from dental_fdi.ui.reference_lines import manual_summary, draw_angle_wedge


class SelectedSectorTests(unittest.TestCase):
    def test_red_marked_lower_sectors_and_signed_names(self):
        guides=[[20,200],[380,200],[200,20],[200,380]]
        for fdi,axis,pixel,angle,name in [
            ('18',[[100,300],[300,100]],(186,206),-45,'Distoangular'),
            ('28',[[100,100],[300,300]],(214,206),45,'Mesioangular')]:
            for ordered in (axis,axis[::-1]):
                for horizontal in (guides[:2],guides[:2][::-1]):
                    points=horizontal+guides[2:]+ordered
                    summary=manual_summary(fdi,points,400,400)
                    self.assertAlmostEqual(summary['signed_horizontal_degrees'],angle)
                    self.assertEqual(summary['screen_category'],name)
                    self.assertLess(summary['X_up_pixels'],0)
                    self.assertEqual(summary['Y_right_pixels']>0,fdi=='28')
                    image=Image.new('RGB',(400,400))
                    draw_angle_wedge(ImageDraw.Draw(image),points,0,image.size,fill='#ffdb35',fdi=fdi)
                    self.assertEqual(image.getpixel(pixel),(255,219,53))
                    self.assertEqual(image.getpixel((200,186)),(0,0,0))
                    self.assertIsNone(summary['angle_to_second_molar_degrees'])

    def test_lower_teeth_keep_previous_upper_ray(self):
        points=[[20,200],[380,200],[200,20],[200,380],[100,300],[300,100]]
        for fdi in ('38','48'):
            summary=manual_summary(fdi,points,400,400)
            self.assertAlmostEqual(summary['signed_horizontal_degrees'],45)
            self.assertGreater(summary['X_up_pixels'],0)
