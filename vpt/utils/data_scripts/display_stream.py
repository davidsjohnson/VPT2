import cv2

from vpt.common import *
from vpt.streams.file_stream import *
import vpt.settings as s

def display_stream(fs):

    i_gen = fs.img_generator()

    for img, fpath in i_gen:

        # img = (ip.normalize(img)*255).astype('uint8')

        cv2.putText(img, fpath, (10, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, .8, (0,0,255))

        cv2.imshow("OG Image", img)
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':

    s.sensor = "realsense"
    folder = "data/rdf/p2/seq_masks"
    ftype ="bmp"
    fs = FileStream(folder, ftype)

    display_stream(fs)