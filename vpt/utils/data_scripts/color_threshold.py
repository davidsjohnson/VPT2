import cv2

from vpt.common import *
import vpt.utils.image_processing as ip
from vpt.streams.file_stream import *

import vpt.settings


def nothing(x):
    pass

def find_color_bounds(fs, participant):

    l_bound = np.array([4, 66, 125])
    u_bound = np.array([25, 255, 255])

    i_gen = fs.img_generator()

    cv2.namedWindow("Image")
    cv2.createTrackbar("Lower H", "Image", l_bound[0], 180, nothing)
    cv2.createTrackbar("Lower S", "Image", l_bound[1], 255, nothing)
    cv2.createTrackbar("Lower V", "Image", l_bound[2], 255, nothing)
    cv2.createTrackbar("Upper H", "Image", u_bound[0], 180, nothing)
    cv2.createTrackbar("Upper S", "Image", u_bound[1], 255, nothing)
    cv2.createTrackbar("Upper V", "Image", u_bound[2], 255, nothing)

    running = True
    for img, fpath in i_gen:

        while(running):

            l_bound[0] = cv2.getTrackbarPos("Lower H", "Image")
            l_bound[1] = cv2.getTrackbarPos("Lower S", "Image")
            l_bound[2] = cv2.getTrackbarPos("Lower V", "Image")

            u_bound[0] = cv2.getTrackbarPos("Upper H", "Image")
            u_bound[1] = cv2.getTrackbarPos("Upper S", "Image")
            u_bound[1] = cv2.getTrackbarPos("Upper V", "Image")

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img_hsv, l_bound, u_bound)

            res = cv2.bitwise_and(img, img, mask=mask)

            cv2.imshow("Image", res)
            k = cv2.waitKey(5)
            if k == ord('n'):
                running = True
                break
            elif k == ord('q'):
                running = False
                break

    cv2.destroyAllWindows()

    print("Lower Color Threshold:", l_bound)
    print("Upper Color Threshold:", u_bound)

    np.save("vpt/utils/data_scripts/lower_bounds_threshold_" + participant, l_bound)
    np.save("vpt/utils/data_scripts/upper_bounds_threshold_" + participant, u_bound)


def run(participant, mask_type):

    og_folder = os.path.join("data/rdf", participant, mask_type + "_masks", "og")
    mask_folder = os.path.join("data/rdf", participant, mask_type + "_masks", "masks")
    ftype = ".bmp"
    fs = FileStream(og_folder, ftype)

    # values from testing with imageJ
    l_bound = np.load("vpt/utils/data_scripts/lower_bounds_threshold_" + participant +".npy")
    u_bound = np.load("vpt/utils/data_scripts/upper_bounds_threshold_" + participant +".npy")

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
            os.makedirs(new_path)

        new_path = os.path.join(new_path, filename)
        cv2.imwrite(new_path, mask)


if __name__ == "__main__":

    folder = "data/rdf/p3/seq_masks"
    # folder = "data/rdf/generated/p0"
    vpt.settings.sensor = 'realsense'
    fs = FileStream(folder, ".bmp")
    # find_color_bounds(fs, "p3")
    #
    run("p3", "seq")