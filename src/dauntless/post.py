import numpy as np

def nms(mag, orient):
    res = np.zeros_like(mag)

    h, w = mag.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            angle = orient[y, x] * 180 / np.pi

            if -112.5 <= angle < -67.5 or 67.5 <= angle < 112.5:
                dx, dy = 0, 1
            elif -67.5 <= angle < -22.5 or 112.5 <= angle < 157.5:
                dx, dy = 1, -1
            elif 22.5 <= angle < 67.5 or -157.5 <= angle < -112.5:
                dx, dy = 1, 1
            else:
                dx, dy = 1, 0

            n1 = mag[y - dy, x - dx]
            n2 = mag[y + dy, x + dx]

            cur = mag[y, x]

            if cur >= n1 and cur >= n2:
                res[y, x] = cur

    return res

def hysteresis(edges, low_t, high_t):
    edges /= np.max(edges)

    strong = edges > high_t
    weak = (edges > low_t) & (edges < high_t)

    res = strong.copy()

    h, w = edges.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not weak[y, x]:
                continue

            if np.any(strong[y - 1:y + 2, x - 1:x + 2]):
                res[y, x] = True

    return res
