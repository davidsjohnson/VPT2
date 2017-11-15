import cv2

from vpt.common import *
from vpt.streams.file_stream import *


def display_stream(fs):

    i_gen = fs.img_generator()

    for img, fpath in i_gen:

        cv2.putText(img, fpath, (10, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, .8, (0,0,255))

        cv2.imshow("OG Image", img)
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':

    folder = "data/rdf/p1/seq_masks"
    ftype ="bmp"
    fs = FileStream(folder, "bmp")

    display_stream(fs)