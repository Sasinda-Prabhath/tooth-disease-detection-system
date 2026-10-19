"""Validated image-coordinate geometry; clinical definitions require review."""
import math


def validate_points(points, count, width, height):
    if len(points) != count:
        raise ValueError(f'{count} landmarks are required')
    for point in points:
        if point is None or len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError('Every landmark needs finite x/y coordinates')
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError('Landmarks must be inside the original image')


def line_angle(reference, target):
    """Acute angle between undirected lines, in [0, 90] degrees."""
    a = [reference[1][i] - reference[0][i] for i in (0, 1)]
    b = [target[1][i] - target[0][i] for i in (0, 1)]
    if min(math.hypot(*a), math.hypot(*b)) < 2:
        raise ValueError('Each line must be at least two pixels long')
    theta = math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    theta = abs(theta)
    return min(theta, 180 - theta)


def measure_reference_lines(points, width, height):
    validate_points(points, 6, width, height)
    return {'red_to_x_degrees': line_angle(points[:2], points[4:6]),
            'red_to_y_degrees': line_angle(points[2:4], points[4:6]),
            'x_to_y_degrees': line_angle(points[:2], points[2:4]),
            'method': 'Acute geometric line angles; not a Winter classification'}


def measure_bone_sites(points, width, height):
    """CEJ/crest/apex per side; Euclidean ratio, no assumed mm or offset."""
    validate_points(points, 6, width, height)
    sites = {}
    for side, offset in [('mesial', 0), ('distal', 3)]:
        cej, crest, apex = points[offset:offset + 3]
        root = math.dist(cej, apex)
        depth = math.dist(cej, crest)
        if root < 2:
            raise ValueError(f'{side}: CEJ-to-apex distance must be at least two pixels')
        root_vector = [apex[i] - cej[i] for i in (0, 1)]
        crest_vector = [crest[i] - cej[i] for i in (0, 1)]
        if sum(a * b for a, b in zip(root_vector, crest_vector)) < 0 or depth > root:
            raise ValueError(f'{side}: crest must be rootward of the CEJ and within the root length; review landmarks')
        sites[side] = {'cej_to_apex_px': root, 'cej_to_crest_px': depth,
                       'radiographic_bone_level_percent': 100 * depth / root}
    return {'sites': sites, 'maximum_percent': max(v['radiographic_bone_level_percent'] for v in sites.values()),
            'definition': '100 * Euclidean CEJ-to-crest / CEJ-to-apex per side; no physiological offset',
            'clinical_stage': 'Not determined from these measurements', 'millimetres': None}
