from multiprocessing import Pool
import random
import time
import cv2
from ultralytics import YOLO

from superflash import RESOURCES_PATH


def work(process_id, duration):
    yolo = YOLO(RESOURCES_PATH / 'yolo11n-pose.pt')
    cap = cv2.VideoCapture(RESOURCES_PATH / 'shop/sample.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    count = 0
    duration = 30
    fps_processing = 10
    dropping_rate = fps_processing / fps
    scale = 1
    imgsz = [frame.shape[0] * scale, frame.shape[1] * scale]
    imgsz = [int(i // 32 * 32) + (32 if i % 32 != 0 else 0) for i in imgsz]
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if not ret or count > duration * fps:
            break
        if random.random() > dropping_rate:
            continue
        yolo(frame, imgsz=imgsz, half=False)



def main():
    parallism = 5
    ts = time.time()
    duration = 30
    with Pool(parallism) as p:
        p.starmap(work, [(i, duration) for i in range(parallism)])
    delay = time.time() - ts
    acc = (duration * parallism) / delay
    print(f'Accelaration: x{acc:.2f}')


if __name__ == "__main__":
    main()