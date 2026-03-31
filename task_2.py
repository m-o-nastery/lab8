import cv2
import numpy as np


if __name__ == "__main__":
    video = cv2.VideoCapture(0)

    while True:
        error, frame = video.read()
        HEIGHT, WIDTH = frame.shape[:2]
        SIZE = 200
        centre_x = WIDTH // 2 - SIZE // 2
        centre_y = HEIGHT // 2 - SIZE // 2

        is_inside = False
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / perimeter**2
            if area > 100 and circularity > 0.7:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if (
                    x >= centre_x
                    and y >= centre_y
                    and x + w <= centre_x + SIZE
                    and y + h <= centre_y + SIZE
                ):
                    is_inside = True
        if is_inside:
            text_color = (0, 255, 0)
        else:
            text_color = (0, 0, 255)
        cv2.putText(
            frame,
            f"{is_inside}",
            (WIDTH - 100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
        )

        cv2.rectangle(
            frame,
            (centre_x, centre_y),
            (centre_x + SIZE, centre_y + SIZE),
            (0, 0, 255),
            2,
        )

        cv2.imshow("video", frame)
        cv2.imshow("thresh", thresh)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
