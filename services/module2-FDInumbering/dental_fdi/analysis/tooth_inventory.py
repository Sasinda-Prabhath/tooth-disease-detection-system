"""Permanent FDI identities and explicit presence-review states; never close gaps."""
from dental_fdi.analysis.dental_analysis import FDI_CODES

FDI_ARCHES = {
    'Upper right': tuple(f'1{i}' for i in range(8, 0, -1)),
    'Upper left': tuple(f'2{i}' for i in range(1, 9)),
    'Lower right': tuple(f'4{i}' for i in range(8, 0, -1)),
    'Lower left': tuple(f'3{i}' for i in range(1, 9)),
}


def validate_missing(codes):
    codes = set(map(str, codes))
    if not codes <= set(FDI_CODES):
        raise ValueError('Missing-tooth review requires permanent FDI codes 11–48')
    return codes


def apply_missing_review(boxes, confirmed_missing=()):
    """Only an explicit review can suppress a detection as confirmed missing."""
    missing = validate_missing(confirmed_missing)
    kept = [box for box in boxes if box['fdi'] not in missing]
    excluded = [box for box in boxes if box['fdi'] in missing]
    return kept, excluded


def tooth_inventory(boxes, confirmed_missing=()):
    missing = validate_missing(confirmed_missing)
    by_fdi = {box['fdi']: box for box in boxes}
    if missing & by_fdi.keys():
        raise ValueError('A confirmed missing tooth cannot also have a displayed detection')
    rows = []
    for fdi in FDI_CODES:
        box = by_fdi.get(fdi)
        status = 'Confirmed missing' if fdi in missing else 'Detected' if box else 'Not detected — review'
        rows.append({'FDI': fdi, 'Status': status,
                     'Evidence': 'User review' if fdi in missing else box.get('source', 'Model prediction') if box else 'No reliable detection',
                     'Score': box.get('conf') if box else None,
                     'Box': box.get('xyxy') if box else None})
    return rows
