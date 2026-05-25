# filename: click_coords_camera.py
import cv2

WINDOW_NAME = "Camera (click to get pixel coords)"

# global state for drawing
click_points = []  # list of (x, y) tuples

def on_mouse(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"Clicked at: x={x}, y={y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # right-click to clear points
        click_points = []
        print("Cleared points")

def main():
    cap = cv2.VideoCapture(0)  # change to 1/2 if you have multiple cameras
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index (1, 2) or check permissions.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Instructions:")
    print(" • Left-click on the video to print/overlay pixel coordinates (x, y)")
    print(" • Right-click to clear overlays")
    print(" • Press 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        # draw markers for all click points
        for (x, y) in click_points:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), thickness=-1)   # filled dot
            cv2.putText(frame, f"({x}, {y})", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
