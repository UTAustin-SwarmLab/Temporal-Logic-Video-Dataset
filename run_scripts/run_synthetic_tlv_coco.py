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
        default=15,
    )
    parser.add_argument(
        "--number_video_per_set_of_frame",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--increase_rate",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ltl_logic",
        type=str,
        default="all",
        choices=[
            "all",
            "F prop1",
            "G prop1",
            "prop1 U prop2",
            "prop1 & prop2",
            "(prop1 & prop2) U prop3",
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

    TLV_generator = SyntheticTLVGenerator(dataloader=dataloader, save_dir=args.save_dir)

    if args.ltl_logic == "all":
        available_tl = [
            "F prop1",
            "G prop1",
            "prop1 U prop2",
            "prop1 & prop2",
            "(prop1 & prop2) U prop3",
        ]  # imagenet does not support & operation.
    else:
        available_tl = [args.ltl_logic]
    for tl in available_tl:
        TLV_generator.generate(
            initial_number_of_frame=args.initial_number_of_frame,
            max_number_frame=args.max_number_frame,
            number_video_per_set_of_frame=args.number_video_per_set_of_frame,
            increase_rate=args.increase_rate,
            ltl_logic=tl,
            save_images=args.save_frames,
        )
