import dauntless

import sys

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

plt.ion()

_, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.pause(0.001)

def main():
    if len(sys.argv) == 2:
        light = Image.open(sys.argv[1]).convert('L')

        dauntless.tags(np.array(light), axs=axs)

        plt.draw()
        plt.pause(0.001)

        while True:
            pass

if __name__ == '__main__':
    main()
