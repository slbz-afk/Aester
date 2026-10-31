"""Union footprints within one exposure, keeping signed RGB residuals.

Input boxes must come from the same frame/master and normalization. Selecting
the greater feather weight selects that pixel's single residual contribution;
selecting the greater RGB signal would instead bias negative noise upward.
"""
import numpy as np


def union_boxes(boxes: list[dict]) -> list[dict]:
    boxes = [b for b in boxes if b["data"].size]
    groups = []
    # Connected bounding boxes avoid a full-frame allocation for distant trails.
    for box in boxes:
        x, y = box["x0"], box["y0"]
        h, w = box["data"].shape[1:]
        group = {"bounds": [x, y, x+w, y+h], "boxes": [box]}
        while True:
            for old in groups:
                a, b = group["bounds"], old["bounds"]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    group["bounds"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    group["boxes"].extend(old["boxes"])
                    groups.remove(old)
                    break
            else:
                break
        groups.append(group)
    out = []
    for group in groups:
        if len(group["boxes"]) == 1:
            out.append(group["boxes"][0])
            continue
        x0, y0, x1, y1 = group["bounds"]
        weights = np.zeros((y1-y0, x1-x0), dtype=np.float64)
        data = np.zeros((3, y1-y0, x1-x0), dtype=np.float32)
        with_validity = ['validity' in b for b in group['boxes']]
        if any(with_validity) and not all(with_validity):
            raise ValueError('cannot combine known and missing same-exposure support')
        validity = np.ones((y1-y0, x1-x0), bool) if all(with_validity) else None
        grid_shapes = [b.get('grid_shape') for b in group['boxes']]
        if any(s is not None for s in grid_shapes) and (
                any(s is None for s in grid_shapes) or any(tuple(s) != tuple(grid_shapes[0]) for s in grid_shapes)):
            raise ValueError('inconsistent same-exposure registered grid dimensions')
        for box in group["boxes"]:
            h, w = box["data"].shape[1:]
            y, x = box["y0"]-y0, box["x0"]-x0
            target_w = weights[y:y+h, x:x+w]
            choose = box["weight"] > target_w
            data[:, y:y+h, x:x+w][:, choose] = box["data"][:, choose]
            np.maximum(target_w, box["weight"], out=target_w)
            if validity is not None:
                source_valid = np.asarray(box['validity'])
                if source_valid.dtype != np.dtype(bool) or source_valid.shape != (h,w):
                    raise ValueError('invalid same-exposure dependency map')
                # Same-exposure disagreement is conservatively unavailable;
                # maximum feather weight must not select a false observation.
                validity[y:y+h, x:x+w] &= (box['weight'] <= 0) | source_valid
        item = {"x0": x0, "y0": y0, "data": data, "weight": weights,
                "footprint_pixels": int(np.count_nonzero(weights))}
        if validity is not None:
            data[:, ~validity] = 0
            item['validity'] = validity
            footprint = weights > 0
            item['source_support'] = {'basis': 'same_exposure_union',
                'footprint_pixels': int(footprint.sum()),
                'supported_pixels': int(np.count_nonzero(footprint & validity)),
                'unsupported_pixels': int(np.count_nonzero(footprint & ~validity))}
        if grid_shapes[0] is not None:
            item['grid_shape'] = tuple(grid_shapes[0])
        out.append(item)
    return sorted(out, key=lambda b: (b["y0"], b["x0"]))
