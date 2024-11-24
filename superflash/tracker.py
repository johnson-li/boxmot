import time
import cv2
import torch
from superflash import PROJECT
from superflash.illustrator import Illustrator
from superflash.model import get_yolo, setup_callbacks
import logging


logger = logging.getLogger(PROJECT)


class Tracker:
    def __init__(self, video_path) -> None:
        self.yolo = get_yolo()
        self.cap = self.capture_video(video_path)
        self.illustrator = Illustrator()

    def capture_video(self, path):
        cap = cv2.VideoCapture(path)
        return cap

    def start_tracking(self):
        i = 0
        offset = (3 * 60 + 40) * 15 + 113
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not self.handle_frame(frame, i):
                break
            i += 1
    
    def handle_frame(self, frame, i) -> bool:
        res = True
        ts = time.time()
        res &= self.handle_frame_tracking(frame, i)
        res &= self.illustrator.handle_frame_illustration(frame, i, self.yolo.predictor.trackers[0])
        delay = time.time() - ts
        logger.info(f'It takes {int(delay * 1000):d} ms to process frame {i}')
        return res

    @torch.no_grad()
    def handle_frame_tracking(self, frame, frame_id) -> bool:
        logger.debug(f'Start tracking frame {frame_id}')
        scale = 1
        imgsz = [frame.shape[0] * scale, frame.shape[1] * scale]
        imgsz = [int(i // 32 * 32) + (32 if i % 32 != 0 else 0) for i in imgsz]
        results = self.yolo.track(
            source=frame,
            conf=.2,
            device='mps',
            # iou=args.iou,
            stream=True,
            imgsz=imgsz,
            # show_conf=args.show_conf,
            classes=[0],
            persist=True
        )
        if frame_id == 0:
            setup_callbacks(self.yolo)
        for result in results:
            boxes = result.boxes
        return True