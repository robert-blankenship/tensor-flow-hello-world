import tensorflow
import numpy
import logging

logging.getLogger().setLevel(logging.DEBUG)


class Network:
    """
    :type minst_data: MinstData
    """
    def __init__(self, minst_data, hidden_nodes=500, activation_function='sigmoid'):
        network = tensorflow.keras.Sequential()

        layers = [
            tensorflow.keras.layers.Dense(minst_data.pixels, activation=activation_function),
            tensorflow.keras.layers.Dense(hidden_nodes, activation=activation_function),
            tensorflow.keras.layers.Dense(10, activation=activation_function)
        ]

        for layer in layers:
            network.add(layer)

        network.compile(optimizer=tensorflow.train.GradientDescentOptimizer(.4),
                        loss=tensorflow.keras.losses.mean_squared_error,
                        metrics=[tensorflow.keras.metrics.categorical_accuracy])

        logging.debug(minst_data.images.shape)
        logging.debug(minst_data.labels.shape)
        network.fit(minst_data.images, minst_data.labels, epochs=10, batch_size=32)

        for i in range(0, 100):
            label = numpy.argmax(minst_data.labels[i])
            actual = network.predict(minst_data.images[i:i+1])
            actual_class = numpy.argmax(actual)
            logging.debug("prediction={}, actual={}".format(label, actual_class))
            logging.debug(minst_data.labels[i])
            logging.debug(actual)


# TODO: Reading all images at once seems to be pretty expensive.
class MinstData:

    def __init__(self, images_path, labels_path, num_images=None):
        with open(labels_path, "r") as labels_file:
            headers = numpy.fromfile(labels_file, dtype='>u4', count=2)

            # Detect the number of images from the file, else, use the user supplied number of images
            if num_images is None:
                self.num_images = int(headers[1])
            else:
                self.num_images = num_images

            labels = numpy.fromfile(labels_file, dtype='>u1', count=self.num_images)

            labels_matrices = []

            for label in labels:
                label_builder = numpy.zeros((10, 1))
                # zero-valued inputs can be problematic for neural networks, so use an arbitrary low value instead.
                label_builder = label_builder + 0.01
                label_builder[label] = 0.99
                labels_matrices.append(label_builder)

            self.labels = numpy.array(labels_matrices)
            self.labels = self.labels.reshape((self.num_images, 10))
            logging.debug(self.labels[0])
            logging.debug(self.labels[1])

        with open(images_path, "r") as images_file:
            # '>i4' corresponds to big endian, unsigned 4 byte integers.
            headers = numpy.fromfile(images_file, dtype='>u4', count=4)

            rows = int(headers[2])
            columns = int(headers[3])
            self.pixels = rows * columns
            logging.debug(self.num_images)
            logging.debug(rows)
            logging.debug(columns)
            logging.debug(self.pixels)

            image1 = numpy.fromfile(images_file, dtype='>u1', count=self.pixels * self.num_images)
            numpy.set_printoptions(linewidth=200)
            self.images = numpy.array(image1.reshape(self.num_images, self.pixels), ndmin=2)

            logging.debug(self.images[0].reshape(28, 28))

            # Scale each element to fit between 0 and 1.
            # Ensure that no element is 0.
            # - "0" values will be -> abs((0/256)-.01) = .01
            # - "256" values will be -> abs(256/256.0 - .01) = .99
            self.images = numpy.absolute(self.images / 256.0 - .01)

            logging.debug(self.images)
            logging.debug(self.images[0].reshape(28, 28))
            logging.debug(self.images.shape)
            logging.debug(self.images[0].shape)


data = MinstData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", num_images=30000)
net = Network(data)
