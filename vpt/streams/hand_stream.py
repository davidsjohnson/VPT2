from vpt.common import *
from vpt.settings import *

class HandStream:

    def __init__(self, filename):

        self._filename = filename
        self.hands = np.load(filename)


    def hand_generator(self):

        for lh_dmaps, lh_label, rh_dmaps, rh_label, filename in zip(self.hands["lh_dmaps"], self.hands["y_lh"],
                                                                    self.hands["rh_dmaps"], self.hands["y_rh"], self.hands["filenames"]):
            yield lh_dmaps, lh_label, rh_dmaps, rh_label, filename