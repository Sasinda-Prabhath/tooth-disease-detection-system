"""Axial radiographic bone-level measurements from six reviewed/predicted landmarks.

YOLO26 landmark adaptation of Lee et al. (2022), not their segmentation model.
The CEJ-to-apex direction is the explicit per-surface reference axis.
"""
import math
from dental_fdi.analysis.manual_geometry import validate_points


def measure_periodontal_sites(points, width, height):
    validate_points(points, 6, width, height)
    sites={}
    for side,offset in [('mesial',0),('distal',3)]:
        cej,crest,apex=points[offset:offset+3]
        vector=[apex[i]-cej[i] for i in (0,1)]
        length=math.hypot(*vector)
        if length<2:
            raise ValueError(f'{side}: CEJ-to-apex axis must be at least two pixels long')
        depth=sum((crest[i]-cej[i])*vector[i] for i in (0,1))/length
        if depth<0 or depth>length:
            raise ValueError(f'{side}: projected crest must be rootward of CEJ and before the apex')
        projection=[cej[i]+depth*vector[i]/length for i in (0,1)]
        sites[side]={'cej_to_apex_px':length,'cej_to_crest_px':depth,
                     'crest_projection':projection,'crest_lateral_offset_px':math.dist(crest,projection),
                     'radiographic_bone_level_percent':100*depth/length}
    return {'sites':sites,'maximum_percent':max(s['radiographic_bone_level_percent'] for s in sites.values()),
            'definition':'100 * axial CEJ-to-crest projection / CEJ-to-apex per surface; no physiological offset',
            'protocol':'yolo26_six_landmark_axial_v1',
            'clinical_stage':'Not assessed; radiographic ratio alone is insufficient',
            'millimetres':None,'normal_bone_level_status':'Unavailable: calibrated spacing and clinical review required',
            'reference':'https://pmc.ncbi.nlm.nih.gov/articles/PMC9026777/'}
