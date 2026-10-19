import math
import unittest
from dental_fdi.analysis.dental_analysis import (
    measure_axes,
    select_detections,
    winter_category,
    classify_winter,
    classify_pell_gregory,
    assess_impaction,
)

class AnalysisTests(unittest.TestCase):
    def test_mesial_and_distal_for_each_quadrant(self):
        for fdi, crown_y, mesial_x in [('18', 100, 100), ('28', 100, -100), ('38', -100, -100), ('48', -100, 100)]:
            for direction in (-1, 1):
                points = [[200, 200], [200+direction*mesial_x, 200+crown_y], [400, 200], [400, 200+crown_y]]
                self.assertAlmostEqual(measure_axes(points, fdi, 600, 600), direction*45)

    def test_reference_axis_not_image_vertical(self):
        self.assertAlmostEqual(measure_axes([[200,200],[250,100],[300,200],[350,100]], '48',600,600), 0)

    def test_horizontal_and_inverted_not_folded(self):
        self.assertAlmostEqual(measure_axes([[200,200],[300,200],[400,200],[400,100]], '48',600,600), 90)
        self.assertAlmostEqual(abs(measure_axes([[200,200],[200,300],[400,200],[400,100]], '48',600,600)), 180)

    def test_literal_published_intervals(self):
        for angle, label in [(-80,'Distoangular'),(-11,'Distoangular'),(-10,'Vertical'),(10,'Vertical'),(11,'Mesioangular'),(80,'Mesioangular'),(81,'Horizontal'),(100,'Horizontal')]:
            self.assertEqual(winter_category(angle, '38'), label)
        for angle in [-180,-80.1,-10.5,10.5,80.5,100.1,180]:
            self.assertIn('review', winter_category(angle, '48'))
        self.assertIn('outside study scope', winter_category(45,'18'))

    def test_clinical_winter_and_pell_gregory(self):
        self.assertEqual(classify_winter(0, '48'), 'Vertical')
        self.assertEqual(classify_winter(45, '48'), 'Mesioangular')
        self.assertEqual(classify_winter(90, '48'), 'Horizontal')
        self.assertEqual(classify_winter(-30, '48'), 'Distoangular')
        self.assertEqual(classify_winter(120, '48'), 'Inverted / Transverse')
        # Pell & Gregory for 48
        depth, ramus = classify_pell_gregory([100, 200, 150, 250], [150, 190, 200, 260], '48')
        self.assertIn('Position', depth)
        self.assertIn('Class', ramus)
        imp = assess_impaction(90, 'Horizontal', depth, ramus)
        self.assertIn('Impacted', imp)

    def test_invalid_points(self):
        for points in [[], [[1,1]]*4, [[1,1],[3,3],[1,1],[float('nan'),3]], [[1,1],[3,3],[1,1],[1000,3]]]:
            with self.assertRaises(ValueError):
                measure_axes(points, '48', 100, 100)

    def test_preserve_classes_bounds_and_adjacent_overlap(self):
        boxes = [dict(pred_fdi='17',conf=.8,xyxy=[10,10,50,90]),
                 dict(pred_fdi='18',conf=.9,xyxy=[10,10,50,90]),
                 dict(pred_fdi='18',conf=.7,xyxy=[11,11,51,91]),
                 dict(pred_fdi='38',conf=.1,xyxy=[20,20,60,80]),
                 dict(pred_fdi='48',conf=.8,xyxy=[70,60,110,120])]
        result = select_detections(boxes, 100, 100)
        self.assertEqual([b['fdi'] for b in result], ['17','18','48'])
        self.assertEqual(result[0]['xyxy'], [10,10,50,90])
        self.assertEqual(result[2]['xyxy'], [70,60,100,100])

    def test_periodontal_bone_loss_formula_and_severity(self):
        from dental_fdi.analysis.dental_analysis import calculate_pbl_ratio, classify_pbl_severity
        from dental_fdi.analysis.periodontal_analysis import classify_furcation_involvement

        # Test ratio formula R = b / a
        self.assertAlmostEqual(calculate_pbl_ratio(100.0, 20.0), 0.20)
        self.assertAlmostEqual(calculate_pbl_ratio(100.0, 50.0), 0.50)
        self.assertAlmostEqual(calculate_pbl_ratio(100.0, 75.0), 0.75)
        self.assertAlmostEqual(calculate_pbl_ratio(0.0, 20.0), 0.0)
        self.assertAlmostEqual(calculate_pbl_ratio(100.0, -5.0), 0.0)

        # Test severity thresholds according to Kao et al. (2025/2026)
        # Normal: <= 15%
        self.assertEqual(classify_pbl_severity(0.10), 'Normal')
        # Mild: <= 1/3 (33.3%)
        self.assertEqual(classify_pbl_severity(0.25), 'Mild')
        self.assertEqual(classify_pbl_severity(1.0 / 3.0), 'Mild')
        # Moderate: 1/3 < R <= 2/3 (66.7%)
        self.assertEqual(classify_pbl_severity(0.40), 'Moderate')
        self.assertEqual(classify_pbl_severity(2.0 / 3.0), 'Moderate')
        # Severe: > 2/3
        self.assertEqual(classify_pbl_severity(0.70), 'Severe')
        self.assertEqual(classify_pbl_severity(0.95), 'Severe')

        # Test molar furcation classification
        furc_normal = classify_furcation_involvement('46', 0.15)
        self.assertFalse(furc_normal['involved'])
        self.assertEqual(furc_normal['grade'], 0)

        furc_mild = classify_furcation_involvement('46', 0.35)
        self.assertTrue(furc_mild['involved'])
        self.assertEqual(furc_mild['grade'], 1)

        furc_incisor = classify_furcation_involvement('11', 0.50)
        self.assertFalse(furc_incisor['involved'])
        self.assertIn('Single-Rooted', furc_incisor['class_name'])


if __name__ == '__main__':
    unittest.main()

