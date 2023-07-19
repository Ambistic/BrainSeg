def align_layers(model_full, model_red):
    layers1 = iter(model_full.layers)
    layers2 = iter(model_red.layers)
    l1 = next(layers1)
    l2 = next(layers2)

    kept_layers1 = []
    kept_layers2 = []
    try:
        while True:
            if l1.name.startswith("low") and not l2.name.startswith("low"):
                l1 = next(layers1)
            else:
                kept_layers1.append(l1)
                kept_layers2.append(l2)
                l1 = next(layers1)
                l2 = next(layers2)
    except StopIteration:
        pass

    return kept_layers1, kept_layers2


def transfer_weights(model_full, model_red):
    layers1, layers2 = align_layers(model_full, model_red)
    for l1, l2 in zip(layers1, layers2):
        l2.set_weights(l1.get_weights())
