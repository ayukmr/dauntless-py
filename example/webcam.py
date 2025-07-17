import dauntless

import cv2
import time

from matplotlib import pyplot as plt

plt.ion()

_, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.pause(0.001)

def main():
    cap = cv2.VideoCapture(1)

    fps = 10
    delay = 1 / fps

    while True:
        start = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        light = hls[:, :, 1]

        dauntless.tags(light, axs=axs)

        plt.draw()
        plt.pause(0.001)

        elapsed = time.time() - start
        time.sleep(max(0, delay - elapsed))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
