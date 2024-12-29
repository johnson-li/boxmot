import cv2
import time


def main():
    rtsp_url = "rtsp://johnson:superflash@192.168.1.100/stream1"

    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        exit()

    # Define the codec and create VideoWriter object
    # 'mp4v' is a codec compatible with MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0  # Adjust based on your camera's frame rate
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_file = "output.mp4"

    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    print("Recording started. Press 'Ctrl+C' to stop.")

    # Record for 1 minute
    start_time = time.time()
    record_duration = 60  # seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to fetch frame")
                break

            # Write the frame to the output file
            out.write(frame)

            # Stop recording after the specified duration
            if time.time() - start_time > record_duration:
                print("Recording completed.")
                break
    except KeyboardInterrupt:
        print("Recording stopped manually.")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()