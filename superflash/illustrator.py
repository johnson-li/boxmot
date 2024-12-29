import colorsys
import hashlib
import cv2
import numpy as np
from boxmot.trackers.basetracker import BaseTracker
from ultralytics.engine.results import Keypoints


class Illustrator:
    def __init__(self) -> None:
        pass

    def handle_frame_illustration(self, frame, frame_id, tracker: BaseTracker, keypoints_list) -> bool:
        img = frame
        # img = tracker.plot_results(img, True)
        # for track in tracker.removed_stracks:
        #     color = [0xff, 0, 0] 
        #     box = track.xyxy
        #     if track.history_observations:
        #         box = track.history_observations[-1]
        #     img = self.plot_box_on_img(img, box, track.conf, track.cls, track.id, 2, .5, color)
        # for track in tracker.lost_stracks:
        #     color = [0, 0xff, 0] if len(track.history_observations) > 2 else [0, 0x7f, 0]
        #     box = track.xyxy
        #     if track.history_observations:
        #         box = track.history_observations[-1]
        #     img = self.plot_box_on_img(img, box, track.conf, track.cls, track.id, 2, .5, color)
        #     img = self.plot_trackers_trajectories(img, track.history_observations, track.id, color)
        for keypoints in keypoints_list:
            keypoints: Keypoints = keypoints.cpu().numpy()
            xy = keypoints.xy
            conf = keypoints.conf
            for i in range(xy.shape[1]):
                x, y = xy[0, i]
                c = conf[0, i]
                if c > 0:
                    img = cv2.circle(img, (int(x), int(y)), 2, [0xff, 0, 0], 2)
        for track in tracker.active_tracks:
            color = self.id_to_color(track.id)
            # color = [0, 0, 0xff]
            box = track.xyxy
            if track.history_observations:
                box = track.history_observations[-1]
            img = self.plot_box_on_img(img, box, track.conf, track.cls, track.id, 2, .5, color)
            img = self.plot_trackers_trajectories(img, track.history_observations, track.id, color)
        cv2.imshow('video', img)
        delay = 1
        if cv2.waitKey(delay) & 0xff == ord('q'):
            return False
        return True

    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5, color=[0, 0, 0xff]) -> np.ndarray:
        img = cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            thickness
        )
        img = cv2.putText(
            img,
            f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
            (int(box[0]), int(box[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            self.id_to_color(id),
            thickness
        )
        return img

    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int, color=[0, 0, 0xff]) -> np.ndarray:
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float (i + 1)) * 1.2)
            img = cv2.circle(
                img,
                (int((box[0] + box[2]) / 2),
                int((box[1] + box[3]) / 2)), 
                2,
                color=color,
                thickness=trajectory_thickness
            )
        return img

    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        
        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]
        
        return bgr
