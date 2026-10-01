import cv2
import numpy as np

def annotate(rgb, teeth, bones=(), impactions=()):
    image = rgb.copy()
    for tooth in teeth:
        cv2.polylines(image, [np.array(tooth.polygon, np.int32)], True, (0, 220, 60), 2)
        x1, y1, x2, y2 = map(int, tooth.box_xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 100, 255), 2)
        label = f'{tooth.fdi} {tooth.confidence:.0%}' + (' ?' if tooth.status == 'uncertain' else '')
        cv2.putText(image, label, (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 100, 255), 1)
    for bone in bones:
        for site in bone.sites:
            for point, color in ((site.cej_point, (255, 190, 0)), (site.crest_point, (0, 200, 255))):
                if point is not None:
                    cv2.circle(image, tuple(map(int, point)), 4, color, -1)
            if site.cej_point is not None and site.crest_point is not None:
                a, b = tuple(map(int, site.cej_point)), tuple(map(int, site.crest_point))
                cv2.line(image, a, b, (255, 190, 0), 2)
                if site.bone_loss_mm is not None:
                    cv2.putText(image, f'{site.bone_loss_mm:.2f}mm', b, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 190, 0), 1)
    for result in impactions:
        tooth = next((t for t in teeth if t.fdi == result.fdi), None)
        if tooth:
            label = 'Uncertain' if result.impacted is None else ('Impacted' if result.impacted else 'Not impacted')
            if result.angulation:
                label += ' / '+result.angulation
            cv2.putText(image, label, (int(tooth.box_xyxy[0]), min(image.shape[0]-5, int(tooth.box_xyxy[3])+16)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 220, 100), 1)
    return image
