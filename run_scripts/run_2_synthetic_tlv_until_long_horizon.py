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
    # - - - - - - Synthetic Generator - - - - - - #
    parser.add_argument(
        "--initial_number_of_frame",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max_number_frame",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--number_video_per_set_of_frame",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--increase_rate",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--ltl_logic",
        type=str,
        default="all",
        choices=["all", "F prop1", "G prop1", "prop1 U prop2", "prop1 & prop2"],
    )
    parser.add_argument(
        "--save_frames",
        type=bool,
        default=False,
    )

    # Start.
    args = parser.parse_args()

    if args.type_of_generator == "synthetic":
        from tlv_dataset.generator.synthetic_tlv_generator import (
            SyntheticTLVGenerator,
        )
        from tlv_dataset.loader.coco import COCOImageLoader
        from tlv_dataset.loader.imagenet import ImageNetDataloader
        from tlv_dataset.loader.cifar import Cifar10ImageLoader
        from tlv_dataset.loader.cifar import Cifar100ImageLoader

        valid_dataloader = ["coco", "imagenet", "cifar10", "cifar100"]
        assert (
            args.dataloader in valid_dataloader
        ), "please use valide dataloader for synthetic tlv dataset: coco, imagenet,cifar10, cifar100"
        if args.dataloader == "coco":
            dataloader = COCOImageLoader()

        elif args.dataloader == "imagenet":
            dataloader = ImageNetDataloader()

        elif args.dataloader == "cifar10":
            dataloader = Cifar10ImageLoader()

        elif args.dataloader == "cifar100":
            dataloader = Cifar100ImageLoader()

        TLV_generator = SyntheticTLVGenerator(
            dataloader=dataloader, save_dir=args.save_dir
        )

        if args.ltl_logic == "all":
            available_tl = [
                "F prop1",
                "G prop1",
                "prop1 U prop2",
                "prop1 & prop2",
            ]
        else:
            available_tl = [args.ltl_logic]
        for tl in available_tl:
            TLV_generator.generate(
                initial_number_of_frame=args.initial_number_of_frame,
                max_number_frame=args.max_number_frame,
                number_video_per_set_of_frame=args.number_video_per_set_of_frame,
                increase_rate=args.increase_rate,
                ltl_logic=tl,
                present_prop1_till_prop2=args.present_prop1_till_prop2,
                save_frames=args.save_frames,
            )
