import cv2

def close_windows():
    """
    Kludge to make destroyAllWindows method work
    """
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)