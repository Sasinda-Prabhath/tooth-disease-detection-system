import unittest
from PIL import Image
from dental_fdi.ui.automatic_angulation import requested_winter_category, render_all_wisdom


class CombinedWisdomTests(unittest.TestCase):
    def test_requested_classification_boundaries(self):
        for value,category in [(-80,'Other / Inverted'),(-79,'Distoangular'),(-11,'Distoangular'),(-10,'Vertical'),(10,'Vertical'),(11,'Mesioangular'),(79,'Mesioangular'),(80,'Horizontal'),(100,'Horizontal'),(101,'Other / Inverted')]:
            self.assertEqual(requested_winter_category(value),category)
        self.assertEqual(requested_winter_category(None),'Unavailable')

    def test_all_four_results_share_one_image(self):
        image=Image.new('RGB',(1000,600),'gray')
        boxes=[];axes={};rows=[]
        for i,fdi in enumerate(('18','28','38','48')):
            x=80+i*220
            boxes.append({'fdi':fdi,'xyxy':[x,150,x+100,350]})
            axes[fdi]={'points':[[x+75,300],[x+25,200]]}
            rows.append({'FDI':fdi,'Angle (degrees)':[0,45,90,-30][i]})
        overlay,results=render_all_wisdom(image,boxes,axes,rows)
        self.assertEqual([r['classification'] for r in results],['Vertical','Mesioangular','Horizontal','Distoangular'])
        self.assertTrue(all(r['horizontal_angle_degrees'] is not None for r in results))
        self.assertGreater(overlay.height,image.height)
        self.assertEqual(image.getpixel((1,1)),(128,128,128))

    def test_missing_reference_and_undetected_tooth_are_unavailable(self):
        image=Image.new('RGB',(400,400))
        boxes=[{'fdi':'38','xyxy':[100,100,200,300]}]
        _,results=render_all_wisdom(image,boxes,{'38':{'points':[[150,250],[120,150]]}},[])
        third=next(r for r in results if r['FDI']=='38')
        self.assertEqual(third['classification'],'Unavailable')
        self.assertIsNotNone(third['horizontal_angle_degrees'])
        self.assertEqual(results[0]['classification'],'Not detected')

    def test_screen_sign_does_not_relabel_anatomical_mesial_tilt(self):
        from dental_fdi.ui.automatic_angulation import screen_axis_tilt
        from dental_fdi.analysis.dental_analysis import measure_axes
        left_axis=[[100,180],[140,100]]
        right_axis=[[300,180],[260,100]]
        self.assertGreater(screen_axis_tilt(left_axis),0)
        self.assertLess(screen_axis_tilt(right_axis),0)
        reference=[[200,180],[200,100]]
        self.assertGreater(measure_axes(left_axis+reference,'48',400,400),0)
        self.assertGreater(measure_axes(right_axis+reference,'38',400,400),0)
