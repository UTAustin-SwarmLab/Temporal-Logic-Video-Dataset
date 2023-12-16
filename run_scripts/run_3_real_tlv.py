import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # - - - - - - - COMMON ARGUMENTS - - - - - - - #
    parser.add_argument(
        "--type_of_generator",
        type=str,
        default="synthetic",
        choices=["synthetic", "real"],
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="coco",
        choices=[
            "coco",
            "imagenet",
            "cifar10",
            "cifar100",
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
        "--verbose",
        type=bool,
        default=False,
    )

    # Start.
    args = parser.parse_args()

    if args.type_of_generator == "real":
        from tlv_dataset.generator.real_tlv_generator import RealTLVGenerator
        from tlv_dataset.loader.nuscenes import NuScenesImageLoader
        from tlv_dataset.loader.waymo import WaymoImageLoader

        valid_dataloader = ["waymo", "nuscenes"]
        assert (
            args.dataloader in valid_dataloader
        ), "please use valide dataloader for real tlv dataset: waymo, nuscenes"

        if args.dataloader == "waymo":
            dataloader = WaymoImageLoader()

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
