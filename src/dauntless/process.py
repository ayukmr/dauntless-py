import numpy as np
from PIL import Image

from . import mask, tags, decode

def process(data, axs=None):
    light = Image.fromarray(data, mode='L')

    w, h = light.size
    scale = 400 / max(w, h)

    light = light.resize(
        (int(w * scale), int(h * scale)),
        Image.Resampling.NEAREST
    )

    img = np.array(light) / 255

    ax = lambda y, x: axs[y][x] if axs is not None else None

    edges = mask.canny(img, ax=ax(0, 0))
    corners = mask.harris(img, ax=ax(0, 1))

    quads = tags.tags(edges, corners, ax=ax(1, 0))

    res = []

    for pts in quads:
        tl = min(pts, key=lambda p:  p[0] + p[1])
        tr = min(pts, key=lambda p: -p[0] + p[1])
        bl = max(pts, key=lambda p: -p[0] + p[1])
        br = max(pts, key=lambda p:  p[0] + p[1])

        corners = (tl, tr, bl, br)
        id = decode.decode(img, corners)

        res.append((id, corners))

    if axs is not None:
        display_tags(axs[1][1], light, res)

    return res

def display_tags(ax, l, tags):
    ax.clear()
    ax.imshow(l, cmap='gray')
    ax.axis('off')

    for id, (tl, tr, bl, br) in tags:
        corners = [tl, tr, br, bl, tl]

        x, y = zip(*corners)
        ax.plot(x, y)

        xs, ys = zip(tl, tr, bl, br)

        x = (min(xs) + max(xs)) / 2
        y = (min(ys) + max(ys)) / 2

        ax.text(
            x, y,
            str(id),
            ha='center',
            va='center',
            c='red',
            fontfamily='Menlo'
        )
