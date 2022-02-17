import cv2

def close_windows():
    """
    Kludge to make destroyAllWindows method work
    """
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

def draw_label(frame, coordinates, label, probability):
    """
    Draws the text label with probability above box face.

    :param frame: frame pixel array
    :param coordinates: tuple with box face coordinates
    :param label: string of predicted label
    :param probability: float of probability
    """
    x1, y1 = coordinates[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.45
    fontColor = (0,255,0)

    text = '%s (%.2f)' % (label[0], probability) 
    cv2.putText(frame, text, (x1, y1 - 10), font, fontScale, fontColor, 1, 2) 