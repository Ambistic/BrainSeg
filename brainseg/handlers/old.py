# just old code that we don't want to delete


def random_shift(low, high=None):
    if high is None:
        high = low
    return choice(range(low, high + 1)) * choice([+1, -1])


def random_angle(low, high=None):
    if high is None:
        high = low
    value = choice(range(low, high + 1))
    sep = choice(range(value + 1))
    x = sep * choice([+1, -1])
    z = (value - sep) * choice([+1, -1])
    return x, z


def get_delta(p_slight=0.5):
    if choices([True, False], [p_slight, 1 - p_slight], k=1)[0]:
        delta_y = random_shift(1)
        delta_angle = random_angle(2)
    else:
        if choice([True, False]):
            delta_y, delta_angle = random_shift(15, 30), (0, 0)
        else:
            delta_y, delta_angle = random_shift(15, 30), (0, 0)
            # delta_y, delta_angle = 0, random_angle(12, 20)

    return delta_y, delta_angle


def enrich_descriptors(descriptors, p=0.5):
    new_descriptors = []
    for desc in descriptors:
        desc = desc.copy()
        delta_y, delta_angle = get_delta(p)

        desc["negative_mri_id"] = desc["mri_id"] + delta_y
        desc["negative_rotx"] = desc["rotx"] + delta_angle[0]
        desc["negative_rotz"] = desc["rotz"] + delta_angle[1]

        desc["positive_mri_id"] = desc["mri_id"]
        desc["positive_rotx"] = desc["rotx"]
        desc["positive_rotz"] = desc["rotz"]

        new_descriptors.append(desc)
    return new_descriptors