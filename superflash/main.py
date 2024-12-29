from superflash import PROJECT, RESOURCES_PATH
from superflash.tracker import Tracker
import logging


def main():
    logging.basicConfig()
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger(PROJECT).setLevel(logging.INFO)
    video_dir = RESOURCES_PATH / 'shop'
    video_path = video_dir / 'sample.mp4'
    # video_path = video_dir / 'test.mp4'
    tracker = Tracker(video_path)
    tracker.set_process_fps(60)
    tracker.start_tracking()
    # input('Press any key to exit...')


if __name__ == "__main__":
    main()