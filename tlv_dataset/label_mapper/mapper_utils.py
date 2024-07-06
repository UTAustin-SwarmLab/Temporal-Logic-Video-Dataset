def get_mapper_metadata(loader_name: str, mapping_to: str):
    loader_name = loader_name.split(".py")[0]
    if loader_name == "imagenet":
        valid_mapper = ["coco"]
        assert (
            mapping_to in valid_mapper
        ), "please use valid mapper for ImageNet: coco"
        if mapping_to == "coco":
            from tlv_dataset.label_mapper.metadata.imagenet_to_coco import (
                MAPPER_METADATA,
            )

            return MAPPER_METADATA
