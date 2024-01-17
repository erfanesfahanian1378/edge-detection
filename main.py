import cv2
import numpy as np

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    mode = 2  # Start with the normal video feed

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny Edge Detection
        edges = cv2.Canny(gray, 100, 200)

        # Dilate the edges to make them thicker
        kernel = np.ones((5,5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            mode = 1  # Edge detection mode
        elif key == ord('2'):
            mode = 2  # Normal video feed
        elif key == ord('3'):
            mode = 3  # Edges overlay on original video

        if mode == 1:
            # Display the edge-detected output
            cv2.imshow('Video Feed', edges)
        elif mode == 2:
            # Display the original video feed
            cv2.imshow('Video Feed', frame)
        elif mode == 3:
            # Overlay edges on original
            edges_colored = np.zeros_like(frame)
            edges_colored[:, :, 2] = edges_dilated  # Red channel
            overlay = cv2.addWeighted(frame, 1, edges_colored, 0.5, 0)
            cv2.imshow('Video Feed', overlay)

        # Exit condition - press 'q' to quit
        if key == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
