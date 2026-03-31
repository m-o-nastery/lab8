import cv2


def main():
    img_bgr = cv2.imread("variant-3.jpeg")
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    cv2.imshow("photo", img_hsv)
    # cv2.imshow("h", h)
    # cv2.imshow("s", s)
    # cv2.imshow("v", v)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
