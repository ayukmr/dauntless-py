import numpy as np
from collections import defaultdict

def tags(edges, corners, ax=None):
    shapes = find_shapes(edges, corners)
    quads = filter_quads(shapes)
    paras = filter_paras(quads)
    res = filter_enclosed(paras)

    if ax:
        ax.clear()
        ax.imshow(edges, cmap='gray')
        ax.axis('off')

        for pts in shapes:
            xs, ys = zip(*pts)
            color = np.random.rand(3)

            ax.scatter(
                xs,
                ys,
                color=color,
                s=5
            )

        ax.text(
            5, 5,
            f'''shapes | {len(shapes)}
 quads | {len(quads)}
 paras | {len(paras)}
-------+---
  tags | {len(res)}''',
            va='top',
            c='red',
            fontfamily='Menlo'
        )

    return res

def find_shapes(edges, corners):
    corners = corners.copy()
    annot = np.zeros_like(edges).astype(np.uint32)

    res = defaultdict(list)
    merge = set()

    h, w = edges.shape
    next = 1

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if not edges[y, x]:
                continue

            patch = annot[y - 2:y + 3, x - 2:x + 3]
            ids = patch[patch != 0]

            if len(ids) > 0:
                if len(ids) > 1:
                    merge |= {
                        (int(i0), int(i1))
                        for i1 in ids
                        for i0 in ids
                        if i0 != i1 and i0 < i1
                    }

                id = ids[0]
            else:
                id = next
                next += 1

            cnrs = np.argwhere(corners[y - 1:y + 2, x - 1:x + 2])

            if len(cnrs) > 0:
                py, px = cnrs[0]
                cx, cy = x - 1 + px, y - 1 + py

                res[id].append((cx, cy))
                corners[cy, cx] = False

            annot[y, x] = id

    for i0, i1 in sorted(merge):
        res[i1] += res[i0]
        del res[i0]

    return [pts for pts in res.values() if pts]

def filter_quads(shapes):
    res = []

    for pts in shapes:
        count = len(pts)

        if count > 4:
            tl = min(pts, key=lambda p:  p[0] + p[1])
            tr = min(pts, key=lambda p: -p[0] + p[1])
            bl = max(pts, key=lambda p: -p[0] + p[1])
            br = max(pts, key=lambda p:  p[0] + p[1])

            outer = [tl, tr, bl, br]

            counted = [*outer]

            for pt in pts:
                if pt in outer:
                    continue

                if not (
                    on_seg(pt, tl, tr)
                    or on_seg(pt, tr, br)
                    or on_seg(pt, br, bl)
                    or on_seg(pt, bl, tl)
                ):
                    counted.append(pt)

            res.append(counted)
        elif count == 4:
            res.append(pts)

    return res

def on_seg(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)

    ab = b - a
    ap = p - a

    norm = np.hypot(ab[0], ab[1])

    if np.cross(ab, ap) > 1.5 * norm:
        return False

    return np.dot(ap, ab) <= np.dot(ab, ab)

def filter_paras(quads):
    res = []

    for pts in quads:
        h = sorted(pts, key=lambda pt: (pt[0], pt[1]))
        v = sorted(pts, key=lambda pt: (pt[1], pt[0]))

        hd0 = v[0][0] - v[1][0]
        hd1 = v[2][0] - v[3][0]

        vd0 = h[0][1] - h[1][1]
        vd1 = h[2][1] - h[3][1]

        if hd0 == 0 or hd1 == 0 or vd0 == 0 or vd1 == 0:
            continue

        h_diff = np.abs(hd0 / hd1)
        v_diff = np.abs(vd0 / vd1)

        if np.abs(h_diff - 1) + np.abs(v_diff - 1) < 0.5:
            res.append(pts)

    return res

def filter_enclosed(quads):
    res = []

    for pts in quads:
        x0s = [pt[0] for pt in pts]
        y0s = [pt[1] for pt in pts]

        x00, x01 = min(x0s), max(x0s)
        y00, y01 = min(y0s), max(y0s)

        enclosed = False

        for others in quads:
            if others == pts:
                continue

            x1s = [pt[0] for pt in others]
            y1s = [pt[1] for pt in others]

            x10, x11 = min(x1s), max(x1s)
            y10, y11 = min(y1s), max(y1s)

            if x10 < x00 and x11 > x01 and y10 < y00 and y11 > y01:
                enclosed = True
                break

        if not enclosed:
            res.append(pts)

    return res
