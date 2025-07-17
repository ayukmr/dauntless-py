import numpy as np

from . import oper, post

HARRIS_K = 0.01
HARRIS_THRESH = 0.2
HARRIS_NEARBY = 3

def canny(img, ax=None):
    blur = oper.blur(img)
    x, y = oper.sobel(blur)

    mag = np.hypot(x, y)
    orient = np.atan2(y, x)

    supp = post.nms(mag, orient)
    mask = post.hysteresis(supp)

    if ax:
        ax.clear()
        ax.imshow(mask, cmap='gray')
        ax.axis('off')

    return mask

def harris(img, ax=None):
    x, y = oper.sobel(img)

    xx = x**2
    yy = y**2
    xy = x * y

    sxx = oper.blur(xx)
    syy = oper.blur(yy)
    sxy = oper.blur(xy)

    det = sxx * syy - sxy**2
    trace = sxx + syy
    resp = det - HARRIS_K * trace**2

    filtered = resp > HARRIS_THRESH
    ys, xs = np.where(filtered)

    mask = np.zeros_like(resp, dtype=bool)

    for y, x in zip(ys, xs):
        x0, x1 = max(0, x - HARRIS_NEARBY), x + HARRIS_NEARBY
        y0, y1 = max(0, y - HARRIS_NEARBY), y + HARRIS_NEARBY

        if resp[y, x] == resp[y0:y1 + 1, x0:x1 + 1].max():
            mask[y, x] = True

    if ax:
        ax.clear()
        ax.imshow(resp, cmap='hot', vmin=0, vmax=1)
        ax.axis('off')

    return mask
