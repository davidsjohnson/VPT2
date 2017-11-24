from vpt.common import *
import vpt.settings as s

import time

# H. Liang, J. Yuan and D. Thalmann, "Parsing the Hand in Depth Images,"
# in IEEE Transactions on Multimedia, vol. 16, no. 5, pp. 1241-1253, Aug. 2014.
#
# http://ieeexplore.ieee.org/abstract/document/6740010/

def generate_feature_offsets(M, radius):
    '''
        Generates offset points in real world space for depth
        context features from a given point.

        M: indicates number of context points
        radius: size of depth context coverage in millimeters
    '''

    feature_offsets = []

    for i in range(-M, M+1):
        for j in range(-M, M+1):

            if i == 0 and j == 0:
                continue

            x = float(j) / M*radius
            y = float(i) / M*radius
            feature_point = [x, y, 0]
            feature_offsets.append(feature_point)

    return feature_offsets


def calc_features(depth_map, offsets, sample_mask = None):

    if sample_mask is None:
        sample_mask = np.ones_like(depth_map, dtype=bool)

    features = depth_map[sample_mask].astype(float)
    features = np.repeat(np.expand_dims(features,1), len(offsets), axis=1)

    points = pixels2points(depth_map, sample_mask)
    points = np.expand_dims(points, 1)
    points = np.repeat(points, len(offsets), axis=1)
    points += offsets

    # TODO:::Make this Faster
    pixels = points2pixels(points)

    # TODO:::Make this Faster
    pixels[pixels < 0] = -1
    pixels[pixels[:, :, 0] >= depth_map.shape[1]] = -1
    pixels[pixels[:, :, 1] >= depth_map.shape[0]] = -1

    features -= depth_map[pixels[:, :, 1], pixels[:, :, 0]]
    features[pixels[:, :, 1] == -1] = 0
    features[pixels[:, :, 0] == -1] = 0

    return features


def points2pixels(points, depth_data=False):

    if s.sensor == "kinect":
        f = 525.5
        px_d = 320
        py_d = 240
        unit = 1000.
        coeffs = None

    elif s.sensor == "realsense":
        f = 476.039
        px_d = 306.79
        py_d = 245.62
        unit = 1000.
        coeffs = (0.127, 0.175, .00334, .00372, -0.115)

    else:
        raise TypeError("Sensor not yet supported")

    # f_ = f/points[:,:,2]

    points_x = points[:,:,0] / points[:,:,2]
    points_y = points[:,:,1] / points[:,:,2]


    # TODO: Implement Distortion: Break it into individual steps...
    # # realsense add back distortion...
    # if s.sensor == "realsense" and coeffs != None:
    #     pass

    pixels = np.zeros_like(points, dtype=int)

    pixels[:, :, 0] = points_x * f + px_d
    pixels[:, :, 1] = points_y * f + py_d

    if not depth_data:
        return pixels[:, :, :2]
    else:
        return pixels[:, :, :2], pixels[:, :, 2:]


def pixels2points(depth_map, sample_mask):

    if s.sensor == "kinect":

        f = 525.5
        px_d = 320.0
        py_d = 240.0
        unit = 1000.
        coeffs = None

    elif s.sensor == "realsense":
        f = 476.039
        px_d = 306.79
        py_d = 245.62
        unit = 1000.
        coeffs = (0.127, 0.175, .00334, .00372, -0.115)

    else:
        raise TypeError("Sensor not yet supported")


    z = depth_map.astype(float) + .00001  # additional term to avoid divide by 0 when converting back to pixels
    z = z[sample_mask]
    z = np.expand_dims(z, axis=0)
    z /= unit      # convert to meters

    pixels = np.indices( depth_map.shape, dtype=float )
    x = pixels[1, :, :][sample_mask]
    y = pixels[0, :, :][sample_mask]

    x = (x - px_d) / f
    y = (y - py_d) / f

    # # Realsense distortion
    # if s.sensor == "realsense" and coeffs != None:
    #     r2 = x*x + y*y
    #     f_ = 1 + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2
    #     x = x * f_ + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x * x)
    #     y = y * f_ + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y * y)

    x = z * x
    y = z * y

    return np.moveaxis(np.concatenate((x, y, z)), 0, 1)