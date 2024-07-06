import argparse

from tlv_dataset.generator.synthetic_tlv_generator import (
    SyntheticTLVGenerator,
)
from tlv_dataset.loader.coco import COCOImageLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # - - - - - - - COMMON ARGUMENTS - - - - - - - #
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--mapping_to",
        type=str,
        default="coco",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
    )
    # - - - - - - Image Loader Arguement - - - - - - #
    parser.add_argument(
        "--coco_image_source",
        type=str,
        default="val",  # train
    )
    # - - - - - - Synthetic Generator - - - - - - #
    parser.add_argument(
        "--initial_number_of_frame",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_number_frame",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--number_video_per_set_of_frame",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--increase_rate",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ltl_logic",
        type=str,
        default="prop1 U prop2",
        choices=[
            "prop1 U prop2",
        ],
    )
    parser.add_argument(
        "--save_images",
        type=bool,
        default=False,
    )

    # Start.
    args = parser.parse_args()

    dataloader = COCOImageLoader(
        coco_root_dir_path=args.data_root_dir,
        coco_image_source=args.coco_image_source,
    )

    TLV_generator = SyntheticTLVGenerator(
        dataloader=dataloader, save_dir=args.save_dir
    )

    TLV_generator.generate_until_time_delta(
        initial_number_of_frame=args.initial_number_of_frame,
        max_number_frame=args.max_number_frame,
        number_video_per_set_of_frame=args.number_video_per_set_of_frame,
        increase_rate=args.increase_rate,
        present_prop1_till_prop2=args.present_prop1_till_prop2,
        save_images=args.save_images,
    )
