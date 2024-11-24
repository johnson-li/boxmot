from superflash import PROJECT, RESOURCES_PATH
from superflash.tracker import Tracker
import logging


def main():
    logging.basicConfig()
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger(PROJECT).setLevel(logging.INFO)
    video_dir = RESOURCES_PATH / 'shop'
    video_path = video_dir / '2.mp4'
    tracker = Tracker(video_path)
    tracker.start_tracking()


if __name__ == "__main__":
    main()