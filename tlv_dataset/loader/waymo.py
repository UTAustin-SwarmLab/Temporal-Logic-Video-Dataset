from __future__ import annotations

import copy

import cv2
from nuscenes.nuscenes import NuScenes

from tlv_dataset.common.utility import save_dict_to_pickle
from tlv_dataset.data import TLVDataset


class WaymoImageLoader:
    """WaymoImageLoader"""

    def __init__(
        self,
    ):
        self.name = "Waymo"

        pass
