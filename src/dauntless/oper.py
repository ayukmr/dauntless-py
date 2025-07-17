import numpy as np

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])
SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
])

def blur(img):
    h, w = img.shape
    y, x = np.ogrid[:h, :w]

    dist = np.hypot(x - w/2, y - h/2)
    sigma = 0.15 * min(h, w)

    freq = to_freq(img)
    mask = np.exp(-(dist**2) / (2 * sigma**2))

    res = freq * mask

    return from_freq(res)

def sobel(img):
    x = apply(img, SOBEL_X)
    y = apply(img, SOBEL_Y)

    return x, y

def apply(img, kernel):
    kh, kw = kernel.shape

    padded = np.zeros_like(img)
    padded[:kh, :kw] = kernel

    f_img = to_freq(img)
    f_knl = to_freq(padded)

    res = f_img * f_knl

    return from_freq(res)

def to_freq(data):
    freq = np.fft.fft2(data)
    shift = np.fft.fftshift(freq)

    return shift

def from_freq(freq):
    shift = np.fft.ifftshift(freq)
    data = np.fft.ifft2(shift)

    return np.real(data)
