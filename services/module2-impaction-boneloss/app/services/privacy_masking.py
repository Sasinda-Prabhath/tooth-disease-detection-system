import cv2
import numpy as np

class PrivacyMasker:
    def __init__(self, root):
        from paddleocr import PaddleOCR
        for name in ('det', 'rec'):
            if not (root / name / 'inference.yml').is_file():
                raise FileNotFoundError('Local PP-OCRv6 det/rec inference exports are required.')
        self.model = PaddleOCR(
            text_detection_model_name='PP-OCRv6_medium_det',
            text_recognition_model_name='PP-OCRv6_medium_rec',
            text_detection_model_dir=str(root / 'det'),
            text_recognition_model_dir=str(root / 'rec'),
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def mask(self, rgb):
        mask = np.zeros(rgb.shape[:2], np.uint8)
        h, w = mask.shape
        warnings = ['OCR masking requires authorised visual review before research reuse.']
        results = list(self.model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
        if len(results) != 1 or 'dt_polys' not in results[0]:
            raise RuntimeError('OCR detection output is missing or incompatible.')
        for result in results:
            # Use all text detections, including low recognition-confidence regions.
            for polygon in result.get('dt_polys', []):
                points = np.asarray(polygon, np.int32)
                x, y, bw, bh = cv2.boundingRect(points)
                if x < .15*w or x+bw > .85*w or y < .20*h or y+bh > .80*h:
                    cv2.fillPoly(mask, [points], 255)
                else:
                    # Do not erase diagnostic anatomy or claim this is deidentified.
                    raise ValueError('Text detected inside diagnostic region; authorised manual privacy review is required.')
        return cv2.dilate(mask, np.ones((7, 7), np.uint8)), warnings
