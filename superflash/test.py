# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import os
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import get_yolo_inferer

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box
from ultralytics.engine.results import Results


RESOURCES_PATH = os.path.join(Path(__file__).parent, 'resources')
DUMP_DIR = os.path.join(RESOURCES_PATH, 'scenario1')
FRAME_SIZE = [2304, 1296]


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    # assert predictor.custom_args.tracking_method in TRACKERS, \
    #     f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    # tracking_method = 'botsort'
    tracking_method = 'bytetrack'
    reid_model = WEIGHTS / 'osnet_x0_25_msmt17.pt'
    tracking_config = TRACKER_CONFIGS / (tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            tracking_method,
            tracking_config,
            reid_model,
            'mps',
            False, # half
            True
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


yolo = YOLO('yolov10n')
# yolo = YOLO('yolov8n')
yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))


@torch.no_grad()
def track(frame, video_out=None):
    scale = 1
    results = yolo.track(
        source=frame,
        conf=.1,
        device='mps',
        # iou=args.iou,
        stream=True,
        imgsz=[1296 * scale, 2304 * scale],
        # show_conf=args.show_conf,
        classes=[0],
        persist=True
    )

    for r in results:
        img = yolo.predictor.trackers[0].plot_results(r.orig_img, True)
        # Check if the human has crossed the door
        # boxes = r.boxes.xyxy.cpu()
        # track_ids = r.boxes.id.int().cpu().tolist()
        # confs = r.boxes.conf.cpu().tolist()
        # for box, track_id, conf in zip(boxes, track_ids, confs):
        #     update_tracing_box(track_id, box[:2], box[2:] + box[:2], conf)

        # font                   = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale              = 1
        # fontColor              = (255,255,255)
        # thickness              = 2
        # lineType               = 2
        # cv2.putText(img, str(TRACKING_DATA['statistics']), 
        #     (10,500), 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     thickness,
        #     lineType)
        # cv2.putText(img, str(TRACKING_DATA['objects']), 
        #     (10,700), 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     thickness,
        #     lineType)

        if video_out:
            video_out.write(img)
        cv2.imshow('BoxMOT', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break


def update_tracing_box(obj_id, top_left, bottom_right, conf):
    window = [
        [927, 131],
        [1294, 1005],
    ]
    # print(obj_id, top_left, bottom_right, conf)
    inside = box_inside(window, [top_left, bottom_right])
    TRACKING_DATA['objects'].setdefault(obj_id, {'inside': inside, 'crossed': False})
    data = TRACKING_DATA['objects'][obj_id]
    if inside != data['inside'] and not data['crossed']:
        data['inside'] = inside
        data['crossed'] = True
        TRACKING_DATA['statistics']['moved out' if inside else 'moved in'] += 1



def open_image(path):
    frame = np.load(path)
    cv2.imshow('image', frame)
    cv2.imwrite(os.path.expanduser('~/Downloads/test.png'), frame)
    key = cv2.waitKey(10000) & 0xFF


TRACKING_DATA = {
    # id => in / out
    'objects': {},
    'statistics': {
        'moved in': 0,
        'moved out': 0,
    },
}


def box_inside(a, b):
    # return a[0][0] <= b[0][0] <= b[1][0] <= a[1][0] and \
    #     a[0][1] <= b[0][1] <= b[1][1] <= a[1][1]
    return b[1][0] <= a[1][0] and b[1][1] <= a[1][1]

def track_all():
    video_out = cv2.VideoWriter(os.path.expanduser('~/Downloads/tracking.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30.0, FRAME_SIZE)
    for path in sorted(os.listdir(DUMP_DIR)):
        path = os.path.join(DUMP_DIR, path)
        if os.path.isfile(path):
            frame = np.load(path)
            # track(frame, video_out)
            track(frame)
    video_out.release()


if __name__ == "__main__":
    track_all()
    # open_image(os.path.join(DUMP_DIR, '0001.npy'))
