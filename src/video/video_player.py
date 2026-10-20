"""
Simple video player for debugging.
"""

from pathlib import Path

import cv2

from src.video.video_reader import VideoReader


def main() -> None:
    video = Path("data/raw/spiel1.mp4")

    reader = VideoReader(video)
    reader.open()

    for frame in reader.frames():

        cv2.imshow("Football Analysis", frame.image)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    reader.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()