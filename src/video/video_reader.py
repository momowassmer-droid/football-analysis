"""
Video reader for football videos.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2

from src.core.frame import Frame
from src.core.video_metadata import VideoMetadata


class VideoReader:
    """Reads football videos frame by frame."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None

    def open(self) -> None:
        """Open the video and read its metadata."""

        self._capture = cv2.VideoCapture(str(self.video_path))

        if not self._capture.isOpened():
            raise FileNotFoundError(
                f"Could not open video: {self.video_path}"
            )

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frame_count / fps if fps else 0.0

        self.metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
        )

    def frames(self) -> Iterator[Frame]:
        """
        Iterate over all frames in the video.
        """

        if self._capture is None:
            raise RuntimeError(
                "Video is not open. Call 'open()' first."
            )

        if self.metadata is None:
            raise RuntimeError(
                "Metadata not available."
            )

        frame_id = 0

        while True:
            success, image = self._capture.read()

            if not success:
                break

            timestamp = frame_id / self.metadata.fps

            yield Frame(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
            )

            frame_id += 1

    def close(self) -> None:
        """Release the video resource."""

        if self._capture is not None:
            self._capture.release()


def main() -> None:
    video = Path("data/raw/spiel1.mp4")

    reader = VideoReader(video)

    reader.open()

    print(reader.metadata)

    print("\nFirst five frames:\n")

    for frame in reader.frames():
        print(
            f"Frame {frame.frame_id:4d} | "
            f"{frame.timestamp:6.2f}s | "
            f"{frame.image.shape}"
        )

        if frame.frame_id >= 4:
            break

    reader.close()


if __name__ == "__main__":
    main()