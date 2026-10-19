import unittest
from PIL import Image
from dental_fdi.ui.automatic_angulation import automatic_reference_geometry, render_automatic_reference


class AutomaticReferenceTests(unittest.TestCase):
    def test_guides_intersect_on_axis_and_angle_uses_horizontal(self):
        points, angle = automatic_reference_geometry([[250,250],[150,100]],400,400)
        self.assertEqual(points[0][1],175)
        self.assertEqual(points[1][1],175)
        self.assertEqual(points[2][0],200)
        self.assertEqual(points[3][0],200)
        self.assertAlmostEqual(angle,56.30993247)

    def test_overlay_renders_for_both_sides_and_preserves_original(self):
        image=Image.new('RGB',(400,400),'black')
        for axis in [[[250,250],[150,100]],[[150,250],[250,100]]]:
            overlay,points,angle=render_automatic_reference(image,[],axis)
            self.assertEqual(overlay.size,image.size)
            self.assertGreater(angle,0)
            self.assertNotEqual(overlay.tobytes(),image.tobytes())
        self.assertEqual(image.getpixel((250,250)),(0,0,0))

    def test_invalid_axis_is_rejected(self):
        for axis in [[[10,10],[10,10]],[[-1,20],[50,50]]]:
            with self.assertRaises(ValueError):
                automatic_reference_geometry(axis,400,400)

    def test_tooth_center_and_mirrored_start_direction(self):
        for fdi,box in [('48',[20,100,100,200]),('38',[300,100,380,200])]:
            for axis in [[[50,160],[90,120]],[[90,120],[50,160]]]:
                points,angle=automatic_reference_geometry(axis,400,400,box,fdi)
                cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
                self.assertEqual(points[0][1],cy)
                self.assertEqual(points[2][0],cx)
                a,b=points[4:]
                self.assertAlmostEqual((cx-a[0])*(b[1]-a[1])-(cy-a[1])*(b[0]-a[0]),0)
                self.assertEqual(a[0]>b[0],fdi=='48')
                self.assertAlmostEqual(angle,45)
