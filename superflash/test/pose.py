from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-pose.pt")
    results = model("https://ultralytics.com/images/bus.jpg")
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen


if __name__ == "__main__":
    main()