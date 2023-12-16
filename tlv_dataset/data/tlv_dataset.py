import dataclasses  # noqa: D100
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from tlv_dataset.common.frame_grouping import combine_consecutive_lists
from tlv_dataset.common.utility import get_file_or_dir_with_datetime


@dataclasses.dataclass
class TLVDataset:
    """TLV Dataset Data Class.

    ground_truth (bool): Ground truth answer of LTL condition for frames
    ltl_frame (str): LTL formula
    number_of_frame (int): Number of frame
    frames_of_interest (list): List of frames that satisfy LTL condition
    - [[0]] -> Frame 0 satisfy LTL condition;
      [[4,5,6,7]] -> Frame 4 to 7 satisfy LTL condition
      [[0],[4,5,6,7]] -> Frame 0 and Frame 4 to 7 satisfy LTL condition.
    labels_of_frame: list of labels of frame
    """

    ground_truth: bool
    ltl_formula: str
    proposition: list
    number_of_frame: int
    frames_of_interest: Optional[List[List[int]]]
    labels_of_frames: List[str]
    images_of_frames: List[np.ndarray] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """Post init."""
        self.frames_of_interest = combine_consecutive_lists(
            data=self.frames_of_interest
        )

    def save_frames(
        self, path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts"
    ) -> None:
        """Save image to path.

        Args:
        path (str, optional): Path to save image.
        """
        from PIL import Image

        for idx, img in enumerate(self.images_of_frames):
            Image.fromarray(img).save(f"{path}/{idx}.png")

    def save(
        self,
        save_path: str = "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts",
    ) -> None:
        """Save the current instance to a pickle file."""
        import pickle

        """Save the current instance to a pickle file."""
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
