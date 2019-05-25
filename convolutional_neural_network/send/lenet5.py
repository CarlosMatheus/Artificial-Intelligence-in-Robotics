from keras import layers, activations
from keras.models import Sequential


def make_lenet5():
    model = Sequential()

    # 1 layer:
    nf = 6
    sx = sy = 1
    fx = fy = 5
    model.add(layers.Conv2D(
        filters=nf,
        kernel_size=(fx, fy),
        strides=(sx, sy),
        activation=activations.tanh,
        input_shape=(32, 32, 1)
        )
    )

    # 2 layer:
    nf = 6
    sx = sy = 2
    # fx = fy = 2
    px = py = 2
    model.add(layers.AveragePooling2D(
        pool_size=(px, py),
        strides=(sx, sy))
    )

    # 3 layer:
    nf = 16
    sx = sy = 1
    fx = fy = 5
    model.add(layers.Conv2D(
        filters=nf,
        kernel_size=(fx, fy),
        strides=(sx, sy),
        activation=activations.tanh,
        input_shape=(14, 14, 1)
        )
    )

    # 4 layer:
    nf = 16
    sx = sy = 2
    # fx = fy = 2
    px = py = 2
    model.add(layers.AveragePooling2D(
        pool_size=(px, py),
        strides=(sx, sy))
    )

    # 5 layer:
    nf = 120
    sx = sy = 1
    fx = fy = 5
    model.add(layers.Conv2D(
        filters=nf,
        kernel_size=(fx, fy),
        strides=(sx, sy),
        activation=activations.tanh,
        input_shape=(5, 5, 1)
    )
    )

    # 6 layer
    num_neurons = 84
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=num_neurons,
        activation=activations.tanh)
    )

    # 7 layer
    num_neurons = 10
    # model.add(layers.Flatten())
    model.add(layers.Dense(
        units=num_neurons,
        activation=activations.softmax)
    )

    return model
