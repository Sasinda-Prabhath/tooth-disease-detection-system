import unittest
from PIL import Image, ImageDraw
from dental_fdi.ui.reference_lines import measure_tooth_reference, draw_angle_wedge


class ReferenceAngleTests(unittest.TestCase):
    def setUp(self):
        self.points = [[10, 50], [90, 50], [50, 10], [50, 90], [20, 20], [80, 80]]

    def test_oblique_axis_and_reversed_endpoints(self):
        for points in [self.points, self.points[:4] + self.points[4:][::-1]]:
            result = measure_tooth_reference(points, 100, 100)
            self.assertAlmostEqual(result['selected_angle_degrees'], 45)
            self.assertAlmostEqual(result['axis_to_vertical_degrees'], 45)
            self.assertAlmostEqual(result['guides_degrees'], 90)

    def test_selected_reference_changes_angle(self):
        points = self.points[:4] + [[10, 70], [90, 70]]
        self.assertEqual(measure_tooth_reference(points, 100, 100, 0)['selected_angle_degrees'], 0)
        self.assertEqual(measure_tooth_reference(points, 100, 100, 2)['selected_angle_degrees'], 90)
        with self.assertRaises(ValueError):
            measure_tooth_reference(points[:4] + [[10, 10], [10, 10]], 100, 100)

    def test_wedge_handles_intersection_parallel_and_off_image(self):
        draw = ImageDraw.Draw(Image.new('RGB', (100, 100)))
        self.assertTrue(draw_angle_wedge(draw, self.points, 0, (100, 100)))
        self.assertFalse(draw_angle_wedge(draw, self.points[:4] + [[10, 70], [90, 70]], 0, (100, 100)))
        self.assertFalse(draw_angle_wedge(draw, self.points[:4] + [[10, 80], [20, 90]], 0, (100, 100)))

    def test_green_sector_is_above_horizontal_for_either_click_order(self):
        points = [[10, 200], [390, 200], [200, 10], [200, 390], [100, 100], [300, 300]]
        for guides in [points[:2], points[:2][::-1]]:
            for axis in [points[4:], points[4:][::-1]]:
                image = Image.new('RGB', (400, 400), 'black')
                self.assertTrue(draw_angle_wedge(ImageDraw.Draw(image), guides + points[2:4] + axis, 0, image.size))
                self.assertEqual(image.getpixel((188, 195)), (105, 207, 32))
                self.assertEqual(image.getpixel((212, 205)), (0, 0, 0))

    def test_horizontal_angle_is_not_vertical_complement(self):
        points = [[10, 200], [390, 200], [200, 10], [200, 390], [100, 50], [300, 350]]
        result = measure_tooth_reference(points, 400, 400)
        self.assertAlmostEqual(result['selected_angle_degrees'], 56.30993247)
        self.assertAlmostEqual(result['axis_to_vertical_degrees'], 33.69006753)
