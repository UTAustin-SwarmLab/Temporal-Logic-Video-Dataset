import argparse

from tlv_dataset.generator.real_tlv_generator import RealTLVGenerator
from tlv_dataset.loader.nuscenes import NuScenesImageLoader
from tlv_dataset.loader.waymo import WaymoImageLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # - - - - - - - COMMON ARGUMENTS - - - - - - - #
    parser.add_argument(
        "--dataloader",
        type=str,
        default="nuscenes",
        choices=[
            "waymo",
            "nuscenes",
        ],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
    )
    # - - - - - - Real Generator - - - - - - #
    parser.add_argument(
        "--unique_propositions",
        type=list,
        default=None,
    )
    parser.add_argument(
        "--tlv_data_dir",
        type=str,
        default=None,
    )
    # - - - - - - Nuscenes Loader - - - - - - #
    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
    )
    parser.add_argument(
        "--mapping_to",
        type=str,
        default="coco",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
    )

    # - - - - - - Waymo Loader - - - - - - #
    parser.add_argument(
        "--config_path",
        type=str,
        default="None",
        help="tlv_dataset/configs/waymo_image_loader.yaml",
    )

    # Start.
    args = parser.parse_args()

    if args.dataloader == "waymo":
        dataloader = WaymoImageLoader(config=args.config_path)

    elif args.dataloader == "nuscenes":
        dataloader = NuScenesImageLoader(
            dataroot=args.dataroot,
            version=args.version,
            verbose=args.verbose,
        )

    TLV_generator = RealTLVGenerator(
        dataloader=dataloader,
        save_dir=args.save_dir,
        unique_propositions=args.unique_propositions,
        tlv_data_dir=args.tlv_data_dir,
    )
    TLV_generator.generate()
