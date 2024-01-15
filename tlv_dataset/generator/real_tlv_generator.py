from __future__ import annotations

import copy
import random
import uuid
from collections import Counter
from pathlib import Path

from _base import DataGenerator

from tlv_dataset.common.utility import load_pickle_to_dict, save_dict_to_pickle
from tlv_dataset.data import TLVDataset


class RealTLVGenerator(DataGenerator):
    def __init__(
        self,
        dataloader,
        save_dir: str,
        unique_propositions: list = None,
        tlv_data_dir: str = None,
    ):
        """LTL ground truth generator.

        Args:
            dataloader (TLVImageLoader): Image data loader.
            save_dir (str): Saving directory
            unique_propositions (List): Unique/Rare propositions from the dataset.
                If given, they will be used for a proposition after "until" operator.
                e.g: non-unique U unique -> [non-unique,non-unique,non-unique,unique].
                Otherwise, the generator will select unique propositions based on the
                data distribution.

        """
        self.tlv_data_dir = tlv_data_dir
        if tlv_data_dir is not None:
            self._tlv_data_dir = Path(tlv_data_dir)
            self._tlv_file_path = list(self._data_dir.glob("*.pkl"))
        else:
            self._tlv_data_dir = None
        self._dataloader = dataloader
        self._unique_propositions = unique_propositions
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def generate(self):
        """Run."""
        if self._tlv_data_dir is not None:
            print(f"Total number of files: {len(self._all_files)}")
            for file in self._all_files:
                benchmark_frame: TLVDataset = load_pickle_to_dict(file)
                self.generate_ltl_ground_truth(benchmark_frame)
        else:
            self._dataloader.loading_data(generate_func=self.generate_ltl_ground_truth)

    def get_label_count(self, lst):
        output = Counter()

        for item in lst:
            if isinstance(item, list):
                # Count occurrences of each element in the sublist
                for sub_item in item:
                    output[sub_item] += 1
            else:
                # Count occurrences of the item
                output[item] += 1

        return dict(output)

    def evaluate_unique_prop(self, unique_prop, lst):
        for item in lst:
            if isinstance(item, list):
                if unique_prop in item:
                    return True
            else:
                if unique_prop == item:
                    return True
                else:
                    return False

    def _class_map(self, prop):
        try:
            map_dict = {
                "vehicle": "car",
                "pedestrian": "person",
                "cyclist": "bicycle",
                "sign": "traffic_sign",
            }
            return map_dict[prop]
        except KeyError:
            return prop

    def f_prop1(self, prop1: str, lst: list[list]):
        ltl_formula = f'F "{self._class_map(prop1)}"'
        new_prop_set = [self._class_map(prop1)]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if prop1 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []
        return ltl_formula, new_prop_set, ltl_ground_truth

    def prop1_u_prop2(self, prop1: str, prop2: str, lst: list[list]):
        ltl_formula = f'"{self._class_map(prop1)}" U "{self._class_map(prop2)}"'
        new_prop_set = [self._class_map(prop1), self._class_map(prop2)]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if prop1 in label:
                    if len(tmp_ground_truth) == 0:
                        tmp_ground_truth.append(idx)
                    else:
                        if prop2 not in label:
                            tmp_ground_truth.append(idx)
            if isinstance(label, list):
                if prop2 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []

        return ltl_formula, new_prop_set, ltl_ground_truth

    def prop1_and_prop2_u_prop3(
        self, prop1: str, prop2: str, prop3: str, lst: list[list]
    ):
        ltl_formula = f'("{self._class_map(prop1)}" & "{self._class_map(prop2)}") U "{self._class_map(prop3)}"'
        new_prop_set = [
            self._class_map(prop1),
            self._class_map(prop2),
            self._class_map(prop3),
        ]
        ltl_ground_truth = []
        tmp_ground_truth = []
        for idx, label in enumerate(lst):
            if isinstance(label, list):
                if all(elem in label for elem in [prop1, prop2]):
                    if len(tmp_ground_truth) == 0:
                        tmp_ground_truth.append(idx)
                    else:
                        if prop3 not in label:
                            tmp_ground_truth.append(idx)

            if isinstance(label, list):
                if prop3 in label:
                    tmp_ground_truth.append(idx)
                    ltl_ground_truth.append(list(set(tmp_ground_truth)))
                    tmp_ground_truth = []

                    # return here since we don't
        return ltl_formula, new_prop_set, ltl_ground_truth

    def get_unique_propositions(self, labels_of_frames: list) -> list:
        # TODO: select unique propositions based on data distribution.
        label_count = self.get_label_count(labels_of_frames)
        sorted_items = sorted(label_count.items(), key=lambda x: x[1])
        try:
            if len(sorted_items) == 3:
                unique_props = [sorted_items[0][0]]  # First two unique values
                highest_count_props = [
                    sorted_items[1][0],
                    sorted_items[2][0],
                ]  # [sorted_items[-1][0], sorted_items[-2][0]]  # First two unique values
            elif len(sorted_items) > 3:
                unique_props = [sorted_items[0][0], sorted_items[1][0]]
                highest_count_props = [sorted_items[2][0], sorted_items[3][0]]
            elif len(sorted_items) < 3:
                if len(sorted_items) == 2:
                    unique_props = [sorted_items[0][0]]
                    highest_count_props = [sorted_items[1][0]]
                else:
                    unique_props = sorted_items[0][0]
                    highest_count_props = sorted_items[0][0]
        except IndexError:
            unique_props = sorted_items[0][0]

        return unique_props

    def generate_ltl_ground_truth(
        self,
        benchmark_frame: TLVDataset,
        unique_prop_threshold: int = 25,
    ):
        ltl_formula = {}
        label_counter = self.get_label_count(benchmark_frame.labels_of_frames)
        if self._unique_propositions is not None:
            unique_propositions = self._unique_propositions
        else:
            unique_propositions = self.get_unique_propositions(
                benchmark_frame.labels_of_frames
            )
        if len(benchmark_frame.proposition) == 0:
            benchmark_frame.proposition = list(
                self.get_label_count(benchmark_frame.labels_of_frames).keys()
            )

        # Start.
        if any(
            unique_prop in benchmark_frame.proposition
            for unique_prop in unique_propositions
        ):
            unique_prop = next(
                (
                    unique_prop
                    for unique_prop in unique_propositions
                    if unique_prop in benchmark_frame.proposition
                ),
                None,
            )

            # F prop
            # if unique props shows up too early, F prop
            if self.evaluate_unique_prop(
                unique_prop,
                benchmark_frame.labels_of_frames[:unique_prop_threshold],
            ):
                ltl_formula, new_prop_set, frames_of_interest = self.f_prop1(
                    prop1=unique_prop, lst=benchmark_frame.labels_of_frames
                )
                self.update_and_save_benchmark_frame(
                    ltl_formula,
                    new_prop_set,
                    frames_of_interest,
                    benchmark_frame,
                    self._save_dir,
                )
            else:
                # prop1 u prop2
                label = benchmark_frame.proposition.copy()
                if unique_prop in label:
                    label.pop(label.index(unique_prop))
                    (
                        ltl_formula,
                        new_prop_set,
                        frames_of_interest,
                    ) = self.prop1_u_prop2(
                        prop1=random.choice(label),
                        prop2=unique_prop,
                        lst=benchmark_frame.labels_of_frames,
                    )
                    self.update_and_save_benchmark_frame(
                        ltl_formula,
                        new_prop_set,
                        frames_of_interest,
                        benchmark_frame,
                        self._save_dir,
                    )
                # prop1_and_prop2_u_prop3
                if len(benchmark_frame.proposition) > 2:
                    label = benchmark_frame.proposition.copy()
                    if unique_prop in label:
                        label.pop(label.index(unique_prop))
                        prop1 = random.choice(label)
                        label.pop(label.index(prop1))
                        prop2 = random.choice(label)
                        (
                            ltl_formula,
                            new_prop_set,
                            frames_of_interest,
                        ) = self.prop1_and_prop2_u_prop3(
                            prop1=prop1,
                            prop2=prop2,
                            prop3=unique_prop,
                            lst=benchmark_frame.labels_of_frames,
                        )
                        self.update_and_save_benchmark_frame(
                            ltl_formula,
                            new_prop_set,
                            frames_of_interest,
                            benchmark_frame,
                            self._save_dir,
                        )

    def label_mapping_function(self, lst):
        new_label = []
        try:
            for item in lst:
                if isinstance(item, list):
                    multi_label = []
                    for item_ in item:
                        multi_label.append(self._class_map(item_))
                    new_label.append(multi_label)
                else:
                    new_label.append(self._class_map(item))
        except KeyError:
            return lst
        return new_label

    def update_and_save_benchmark_frame(
        self,
        ltl_formula,
        new_prop_set,
        frames_of_interest,
        benchmark_frame: TLVDataset,
        save_dir,
    ):
        file_name = f"benchmark_{self._dataloader.name}_ltl_{ltl_formula}_{len(benchmark_frame.images_of_frames)}_{uuid.uuid4()}.pkl"
        benchmark_frame_ = copy.deepcopy(benchmark_frame)
        benchmark_frame_.frames_of_interest = frames_of_interest
        benchmark_frame_.ltl_formula = ltl_formula
        benchmark_frame_.proposition = new_prop_set
        benchmark_frame_.labels_of_frames = self.label_mapping_function(
            benchmark_frame.labels_of_frames
        )
        save_dict_to_pickle(
            dict_obj=benchmark_frame_, path=save_dir, file_name=file_name
        )
