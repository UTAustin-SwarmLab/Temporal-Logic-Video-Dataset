MAPPER_METADATA = {
    "animal": None,  # Assuming a generic animal category
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.personal_mobility": "person",
    "human.pedestrian.police_officer": "person",
    "human.pedestrian.stroller": "person",
    "human.pedestrian.wheelchair": "person",
    "movable_object.barrier": None,  # No direct equivalent
    "movable_object.debris": None,  # No direct equivalent
    "movable_object.pushable_pullable": None,  # No direct equivalent
    "movable_object.trafficcone": None,  # No direct equivalent
    "static_object.bicycle_rack": None,  # Possibly 'bicycle' if the bikes are significant in the image
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "car",  # Broadly categorizing as 'truck'
    "vehicle.emergency.ambulance": "car",  # Or 'car', depending on size/shape
    "vehicle.emergency.police": "car",
    "vehicle.motorcycle": "bicycle",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
    # Additional mappings for other labels
    "flat.driveable_surface": None,  # No direct equivalent
    "flat.other": None,  # No direct equivalent
    "flat.sidewalk": None,  # No direct equivalent
    "flat.terrain": None,  # No direct equivalent
    "static.manmade": None,  # No direct equivalent, but could consider 'building' if part of your labels
    "static.other": None,  # No direct equivalent
    "static.vegetation": None,  # Could be 'tree' or 'plant' if those are part of your labels
    "vehicle.ego": "car",  # Assuming this is the ego vehicle (the car with the sensor/camera)
    "noise": None,  # No direct equivalent
}
