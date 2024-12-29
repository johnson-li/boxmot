
from functools import partial
from pathlib import Path
import torch
from ultralytics import YOLO

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS, WEIGHTS
from superflash import RESOURCES_PATH


def on_predict_start(predictor, persist=False):
    if predictor.trackers:
        return
    # tracking_method = 'bytetrack'
    tracking_method = 'botsort'
    reid_model = RESOURCES_PATH / 'osnet_ain_x1_0_msmt17.pt'
    tracking_config = TRACKER_CONFIGS / (tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            tracker_type=tracking_method,
            tracker_config=tracking_config,
            reid_weights=reid_model,
            device='mps',
            half=True,
            per_class=False,
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes.data).cpu().numpy()
        print(det)
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def get_yolo():
    yolov10_models = ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10b.pt', 'yolov10l.pt', 'yolov10x.pt']
    # yolo = YOLO(RESOURCES_PATH / yolov10_models[0])
    yolo = YOLO(RESOURCES_PATH / 'yolo11n-pose.pt')
    # on_predict_start(yolo.predictor, True)
    return yolo


def setup_callbacks(yolo):
    yolo.predictor.trackers = None
    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    # yolo.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=True))
