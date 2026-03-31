import cv2
import numpy as np


def main():
    video = cv2.VideoCapture(0)
    fly = cv2.imread("fly64.png")

    while True:
        error, frame = video.read()
        height, width = frame.shape[:2]
        size = 200
        centre_x = width // 2 - size // 2
        centre_y = height // 2 - size // 2

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

                point_x = x + w // 2
                point_y = y + h // 2

                fly_h, fly_w = fly.shape[:2]
                y1 = point_y - fly_h // 2
                y2 = y1 + fly_h
                x1 = point_x - fly_w // 2
                x2 = x1 + fly_w
                frame[y1:y2, x1:x2] = fly

                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if (
                    x >= centre_x
                    and y >= centre_y
                    and x + w <= centre_x + size
                    and y + h <= centre_y + size
                ):
                    is_inside = True

        if is_inside:
            text_color = (0, 255, 0)
        else:
            text_color = (0, 0, 255)

        cv2.putText(
            frame,
            f"{is_inside}",
            (width - 100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
        )

        cv2.rectangle(
            frame,
            (centre_x, centre_y),
            (centre_x + size, centre_y + size),
            (0, 0, 255),
            2,
        )

        cv2.imshow("video", frame)
        cv2.imshow("thresh", thresh)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
