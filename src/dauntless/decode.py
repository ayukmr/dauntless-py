import numpy as np

CODES = [
    57401312644,
    58383764297,
    59366215950,
    61331119256,
    63296022562,
    65260925868,
    1453707397,
    4401062356,
    9313320621,
    10295772274,
    14225578886
]

def decode(img, corners):
    tag = sample(img, corners)
    bits = (tag / tag.max()) > 0.5

    for r in range(4):
        rot = np.rot90(bits, k=r)

        bin = ''.join(
            str(int(b))
            for b in rot.reshape(-1)
        )

        id = int(bin, 2)

        if id in CODES:
            return CODES.index(id)

def sample(img, corners):
    idx = (np.arange(6) + 1 + 0.5) / 8
    u, v = np.meshgrid(idx, idx)

    tl, tr, bl, br = corners

    ix = (
        (1 - u) * (1 - v) * tl[0] +
        u * (1 - v) * tr[0] +
        u * v * br[0] +
        (1 - u) * v * bl[0]
    ).astype(int)

    iy = (
        (1 - u) * (1 - v) * tl[1] +
        u * (1 - v) * tr[1] +
        u * v * br[1] +
        (1 - u) * v * bl[1]
    ).astype(int)

    return img[iy, ix]
