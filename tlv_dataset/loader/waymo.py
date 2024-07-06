from __future__ import annotations

import io
import os
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple

import dask.dataframe as dd
import numpy as np
import omegaconf
import PIL.Image as Image
import tensorflow as tf

from tlv_dataset.data import TLVDataset

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

import os
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2

PKG_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent
CONFIG_PATH = PKG_PATH / "config" / "waymo_image_loader.yaml"


def read(config, tag: str, context_name: str, validation=False) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    if validation:
        paths = f"{config.EVAL_DIR}/{tag}/{context_name}.parquet"
    else:
        paths = f"{config.TRAIN_DIR}/{tag}/{context_name}.parquet"

    try:
        df = dd.read_parquet(paths)
    except:
        raise ValueError(f"Could not read {paths}")
    return df


def ungroup_row(
    key_names: Sequence[str], key_values: Sequence[str], row: dd.DataFrame
) -> Iterator[Dict[str, Any]]:
    """Splits a group of dataframes into individual dicts."""
    keys = dict(zip(key_names, key_values))
    cols, cells = list(zip(*[(col, cell) for col, cell in row.items()]))
    for values in zip(*cells):
        yield dict(zip(cols, values), **keys)


def load_data_set_parquet(
    config: omegaconf,
    context_name: str,
    validation=False,
    context_frames: List = None,
) -> Tuple[
    List[open_dataset.CameraImage], List[open_dataset.CameraSegmentationLabel]
]:
    """Load datset from parquet files for segmentation and camera images

    Args:
        config: OmegaConf DictConfig object with the data directory and file name (see config,yaml)

    Returns:
       cam_segmentation_list: List of segmentation labels ordered by the camera order
    """
    cam_box_df = read(config, "camera_box", context_name)
    cam_images_df = read(config, "camera_image", context_name)

    merged_df = v2.merge(cam_images_df, cam_box_df, right_group=True)

    # Group segmentation labels into frames by context name and timestamp.
    frame_keys = ["key.segment_context_name", "key.frame_timestamp_micros"]

    if context_frames is None:
        # frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
        cam_segmentation_per_frame_df = merged_df.groupby(
            frame_keys, group_keys=False
        ).agg(list)
    else:
        # frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
        # cam_segmentation_per_frame_df = merged_df.groupby(
        #     frame_keys, group_keys=True).agg(list)

        # filter out the frames that are not in the context_frames
        cam_segmentation_per_frame_df = merged_df.reset_index()
        cam_segmentation_per_frame_df = cam_segmentation_per_frame_df.set_index(
            "key.frame_timestamp_micros"
        )
        cam_segmentation_per_frame_df = cam_segmentation_per_frame_df.loc[
            context_frames
        ]
        cam_segmentation_per_frame_df = cam_segmentation_per_frame_df.groupby(
            frame_keys, group_keys=False
        ).agg(list)

    cam_box_list = []
    image_list = []
    for i, (key_values, r) in enumerate(
        cam_segmentation_per_frame_df.iterrows()
    ):
        # Read three sequences of 5 camera images for this demo.
        # Store a segmentation label component for each camera.
        cam_box_list.append(
            [
                v2.CameraBoxComponent.from_dict(d)
                for d in ungroup_row(frame_keys, key_values, r)
            ]
        )
        image_list.append(
            [
                v2.CameraImageComponent.from_dict(d)
                for d in ungroup_row(frame_keys, key_values, r)
            ]
        )

    # TODO: need to figure out what the function is to obtain camera images
    # for i, (key_values, r) in enumerate(cam_images_per_frame_df.iterrows()):
    #     # Read three sequences of 5 camera images for this demo.
    #     # Store a segmentation label component for each camera.
    #     cam_list.append(
    #         [v2.CameraSegmentationLabelComponent.from_dict(d)
    #         for d in ungroup_row(frame_keys, key_values, r)])

    return cam_box_list, image_list


def read_box_labels(
    config: omegaconf, box_labels: List[v2.CameraBoxComponent]
) -> Tuple[List, List, List]:
    """The dataset provides tracking for instances between cameras and over time.
    By setting remap_to_global=True, this function will remap the instance IDs in
     each image so that instances for the same object will have the same ID between
     different cameras and over time.

    Args:
        config: omega congif gfrom the config.yaml file
        segmentation_protos_ordered: List of segmentation labels ordered by the camera order

    Returns:
        box_class: List of object types (classes)
        bounding_boxes: List of bounding boxes
    """
    # We can further separate the semantic and instance labels from the panoptic
    # labels.
    NUM_CAMERA_FRAMES = 5
    box_classes_frame = [
        [] for _ in range(NUM_CAMERA_FRAMES)
    ]  # For the entire frame per camera
    bounding_boxes_frame = [
        [] for _ in range(NUM_CAMERA_FRAMES)
    ]  # For the entire frame per camera
    try:
        for i in range(0, len(box_labels)):
            for j in range(len(box_labels[i])):
                # Get camera ID
                cam_id = box_labels[i][j].key.camera_name - 1
                box_classes_frame[cam_id].append(box_labels[i][j].type)
                bounding_boxes_frame[cam_id].append(
                    [
                        np.array(
                            [
                                box_labels[i][j].box.center.x,
                                box_labels[i][j].box.center.y,
                                box_labels[i][j].box.size.x,
                                box_labels[i][j].box.size.y,
                            ]
                        )
                    ]
                )
    except:
        print("Box labels not found")
        box_classes_frame = None
        bounding_boxes_frame = None

    return box_classes_frame, bounding_boxes_frame


def read_camera_images(
    config: omegaconf, camera_images: List[v2.CameraImageComponent]
) -> List[np.ndarray]:
    """Read camera images from the dataset

    Args:
        config: omega config from the config.yaml file
        camera_images: List of camera images

    Returns:
        camera_images: List of camera images
    """
    NUM_CAMERA_FRAMES = 5
    camera_images_all = []

    for i in range(0, len(camera_images)):
        camera_images_frame = [[] for _ in range(NUM_CAMERA_FRAMES)]
        for j in range(len(camera_images[i])):
            cam_id = camera_images[i][j].key.camera_name - 1
            camera_images_frame[cam_id] = np.array(
                Image.open(io.BytesIO(camera_images[i][j].image))
            )
        camera_images_all.append(camera_images_frame)
    return camera_images_all


class WaymoImageLoader(Dataset):
    """Loads the dataset from the pickled files"""

    CLASSES: str
    PALLETE: str
    FOLDER: str
    # camera_files: List[str]
    # segment_files: List[str]
    # instance_files: List[str]
    num_images: int
    context_set: Set[str]
    segment_frames: Dict[str, List[str]]
    ds_length: int
    ds_config: omegaconf

    def __init__(self, config=None, validation=False) -> None:
        super().__init__()
        if config is None:
            config = omegaconf.OmegaConf.load(CONFIG_PATH)
        else:
            config = omegaconf.OmegaConf.load(config)
        if validation:
            self.FOLDER = config.EVAL_DIR
            self.contexts_path = os.path.join(
                self.FOLDER, "2d_detection_validation_metadata.txt"
            )
        else:
            self.FOLDER = config.TRAIN_DIR
            self.contexts_path = os.path.join(
                self.FOLDER, "2d_detection_training_metadata.txt"
            )

        self.context_set = set()
        self.segment_frames = dict()
        self.num_images = 0
        self.ds_config = config
        self.validation = validation
        self.context_count = dict()
        self.max_context_cam = dict()

        with open(self.contexts_path) as f:
            for line in f:
                context_name = line.strip().split(",")[0]
                context_frame = line.strip().split(",")[1]
                self.context_set.add(context_name)
                if self.context_count.get(context_name) is None:
                    self.context_count[context_name] = {
                        (j + 1): 0 for j in range(5)
                    }
                camera_ids = []
                available_camera_ids = line.strip().split(",")[2:-1]
                for camera_id in available_camera_ids:
                    self.context_count[context_name][int(camera_id)] += 1
                    camera_ids.append(int(camera_id))

        for k, v in self.context_count.items():
            self.max_context_cam[k] = max(v, key=v.get)

        self.ds_length = len(self.context_set)
        # assert len(self.camera_files) == len(self.segment_files)\
        #       == len(self.instance_files), \
        #     "The number of files in the camera, segmentation and instance folders \
        # are not equal"

        # self.num_images = len(self.camera_files)
        # # Find the number of images

        # RGB colors used to visualize each semantic segmentation class.
        self.CLASSES_TO_PALLETTE = {
            "undefined": [0, 0, 0],
            "ego_vehicle": [102, 102, 102],
            "car": [0, 0, 142],
            "truck": [0, 0, 70],
            "bus": [0, 60, 100],
            "other_large_vehicle": [61, 133, 198],
            "bicycle": [119, 11, 32],
            "motorcycle": [0, 0, 230],
            "trailer": [111, 168, 220],
            "pedestrian": [220, 20, 60],
            "cyclist": [255, 0, 0],
            "motorcyclist": [180, 0, 0],
            "bird": [127, 96, 0],
            "ground_animal": [91, 15, 0],
            "construction_cone_pole": [230, 145, 56],
            "pole": [153, 153, 153],
            "pedestrian_object": [234, 153, 153],
            "sign": [246, 178, 107],
            "traffic_light": [250, 170, 30],
            "building": [70, 70, 70],
            "road": [128, 64, 128],
            "lane_marker": [234, 209, 220],
            "road_marker": [217, 210, 233],
            "sidewalk": [244, 35, 232],
            "vegetation": [107, 142, 35],
            "sky": [70, 130, 180],
            "ground": [102, 102, 102],
            "dynamic": [102, 102, 102],
            "static": [102, 102, 102],
        }

        self.OBJECT_CLASSES = [
            "undefined",
            "vehicle",
            "pedestrian",
            "sign",
            "cyclist",
        ]

        self.CLASSES = list(self.CLASSES_TO_PALLETTE.keys())
        self.PALLETE = list(self.CLASSES_TO_PALLETTE.values())
        self.color_map = np.array(self.PALLETE).astype(np.uint8)

    def __len__(self) -> int:
        # return max(len(self.camera_files),
        #             len(self.segment_files),
        #             len(self.instance_files))
        return self.ds_length

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Returns an item from the dataset referenced by index

        Args:
            index: The index of the item to return
        Returns:
            camera_images: The camera images
            semantic_mask_rgb: The semantic mask in rgb format
            instance_masks: The instance masks
            object_masks: The object masks
            img_data (dict): The image data
        """

        if index >= self.ds_length:
            raise IndexError("Index out of range")

        # Find the appropriate index at which the image is stored
        index_copy = index
        context_name = list(self.context_set)[index_copy]

        # Load all the frames from the context file
        frames_with_box, camera_images = load_data_set_parquet(
            config=self.ds_config,
            context_name=context_name,
            validation=self.validation,
        )

        class_types, _ = read_box_labels(self.ds_config, frames_with_box)

        camera_images_frame = read_camera_images(self.ds_config, camera_images)
        camera_id = self.max_context_cam[context_name]
        # All semantic labels are in the form of object indices defined by the PALLETE
        # flatten list and remvoe empty sublists
        camera_images_frame = [
            x[camera_id - 1]
            for x in camera_images_frame
            if x[camera_id - 1] != []
        ]
        class_types_frame = class_types[camera_id - 1]
        class_types_frame = [x for x in class_types_frame if x != []]

        for i in range(len(class_types_frame)):
            for j in range(len(class_types_frame[i])):
                class_types_frame[i][j] = self.OBJECT_CLASSES[
                    class_types_frame[i][j]
                ]
            class_types_frame[i] = list(set(class_types_frame[i]))
        return camera_images_frame, class_types_frame

    def loading_data(self, generate_func: callable = None):
        """Loads the data from the dataset

        Args:
            generate_func: The function to generate the data
        """
        n_vids = 0
        dataset = self
        for i in tqdm(range(len(dataset))):
            frames, objects = dataset[i]
            object_flat = sum(objects, [])
            if len(set(object_flat)) > 2:
                waymo_ltl_frame = TLVDataset(
                    number_of_frame=len(frames),
                    images_of_frames=frames,
                    labels_of_frames=objects,
                    ground_truth=True,
                    ltl_formula="",
                    proposition=list(set(object_flat)),
                    frames_of_interest=[],
                )
                generate_func(waymo_ltl_frame)
                n_vids += 1


if __name__ == "__main__":
    # config = omegaconf.OmegaConf.load("ns_vfs/loader/config.yaml")
    dataset = WaymoImageLoader()

#     store_pickle_folder = "/store/nsvs_artifact/waymo_benchmark_video"
# n_vids = 0
# for i in tqdm(range(len(dataset))):
#     frames, objects = dataset[i]
#     object_flat = sum(objects, [])
#     if len(set(object_flat)) > 2:
#         waymo_ltl_frame = TLVDataset(
#             number_of_frame=len(frames),
#             images_of_frames=frames,
#             labels_of_frames=objects,
#             ground_truth=True,
#             ltl_formula="",
#             proposition=list(set(object_flat)),
#             frames_of_interest=[],
#         )
#         save_dict_to_pickle(
#             waymo_ltl_frame,
#             store_pickle_folder,
#             f"waymo_benchmark_video_{n_vids}.pkl",
#         )
#         n_vids += 1
