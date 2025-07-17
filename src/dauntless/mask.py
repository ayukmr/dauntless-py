import numpy as np

from . import oper, post

def canny(img, ax=None):
    blur = oper.blur(img)
    x, y = oper.sobel(blur)

    mag = np.hypot(x, y)
    orient = np.atan2(y, x)

    supp = post.nms(mag, orient)
    mask = post.hysteresis(supp, 0.05, 0.2)

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
    resp = det - 0.01 * trace**2

    filtered = resp > 0.2
    ys, xs = np.where(filtered)

    mask = np.zeros_like(resp, dtype=bool)

    for y, x in zip(ys, xs):
        x0, x1 = max(0, x - 3), x + 3
        y0, y1 = max(0, y - 3), y + 3

        if resp[y, x] == resp[y0:y1 + 1, x0:x1 + 1].max():
            mask[y, x] = True

    if ax:
        ax.clear()
        ax.imshow(resp, cmap='hot', vmin=0, vmax=1)
        ax.axis('off')

    return mask
