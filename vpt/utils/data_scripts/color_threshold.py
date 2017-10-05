import cv2

from vpt.common import *
import vpt.utils.image_processing as ip
from vpt.streams.file_stream import *

def run():

    og_folder = "data/rdf/p4/cae_masks/og"
    mask_folder = "data/rdf/p4/cae_masks/masks"
    ftype = ".bmp"
    fs = FileStream(og_folder, ftype)

    # values from testing with imageJ
    l_bound = np.array([4, 66, 125])
    u_bound = np.array([25, 255, 255])

    iGen = fs.img_generator()

    for img, fname in iGen:

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, l_bound, u_bound)
        # mask = cv2.GaussianBlur(mask, (3,3), 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        temp = fname.split("/")
        filename = temp[-1].strip(ftype) + ".jpg"
        exercise = temp[-2]
        new_path = os.path.join(mask_folder, exercise)

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        new_path = os.path.join(new_path, filename)
        cv2.imwrite(new_path, mask)


if __name__ == "__main__":

    run()