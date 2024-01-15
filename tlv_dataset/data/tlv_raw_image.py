import dataclasses  # noqa: D100
import random
from typing import List

import numpy as np
import torch


@dataclasses.dataclass
class TLVRawImage:
    """Benchmark image frame class."""

    unique_labels: list
    labels: List[List[str]]
    images: List[np.ndarray]

    def sample_image_from_label(self, labels: list, proposition: list) -> np.ndarray:
        """Sample image from label."""
        image_of_frame = []
        img_to_label = {}
        img_to_label_list = {}
        for prop in proposition:
            # img_to_label[prop] = [i for i, value in enumerate(self.labels) if value == prop]
            img_to_label[prop] = [
                i for i, value in enumerate(self.labels) if prop in value
            ]
        # img_to_label_list[tuple(sorted(proposition))] = [
        #     i for i, value in enumerate(self.labels) if all(prop in value for prop in proposition)
        # ]

        label_idx = 0
        for label in labels:
            if label is None:
                while True:
                    random_idx = random.randrange(
                        len(self.images)
                    )  # pick one random image with idx
                    val = []
                    for single_label in self.labels[
                        random_idx
                    ]:  # go over all labels of the image
                        if single_label in proposition:
                            val.append(True)
                    if True not in val:
                        labels[label_idx] = single_label
                        image_of_frame.append(self.images[random_idx])
                        break
            else:
                # lable available - just get the image
                if isinstance(label, str):
                    # one label in the frame
                    random_idx = random.choice(img_to_label[label])
                    # plt.imshow(self.images[random_idx])
                    # plt.axis("off")
                    # plt.savefig("data_loader_sample_image.png")
                    image_of_frame.append(self.images[random_idx])
                elif isinstance(label, list):
                    img_to_label_list[tuple(sorted(label))] = [
                        i
                        for i, value in enumerate(self.labels)
                        if all(prop in value for prop in label)
                    ]
                    random_idx = random.choice(img_to_label_list[tuple(sorted(label))])
                    image_of_frame.append(self.images[random_idx])
            label_idx += 1
        return labels, image_of_frame


@dataclasses.dataclass
class TLVRawImageDataset:
    """Benchmark image frame class with a torchvision dataset for large datasets."""

    unique_labels: list
    labels: List[List[str]]
    images: torch.utils.data.Dataset

    def sample_image_from_label(self, labels: list, proposition: list) -> np.ndarray:
        """Sample image from label."""
        image_of_frame = []
        img_to_label = {}
        for prop in proposition:
            # img_to_label[prop] = [i for i, value in enumerate(self.labels) if value == prop]
            img_to_label[prop] = [
                i for i, value in enumerate(self.labels) if prop in value
            ]

        label_idx = 0
        for label in labels:
            if label is None:
                while True:
                    random_idx = random.randrange(len(self.images))
                    if self.labels[random_idx] not in proposition:
                        break
                labels[label_idx] = self.labels[random_idx]
                image_of_frame.append(self.images[random_idx][0])
            else:
                random_idx = random.choice(img_to_label[label])
                image_of_frame.append(self.images[random_idx][0])

            label_idx += 1
        return labels, image_of_frame
