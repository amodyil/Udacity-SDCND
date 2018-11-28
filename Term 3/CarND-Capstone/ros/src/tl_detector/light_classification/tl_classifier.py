from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os
import cv2


class TLClassifier(object):
    def __init__(self, is_site):
        if is_site:
            file = '/frozen_inference_graph_site.pb'
        else:
            file = '/frozen_inference_graph_sim.pb'

        graph_file = os.path.dirname(os.path.realpath(__file__)) + file
        graph = self.load_graph(graph_file)

        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.boxes = graph.get_tensor_by_name('detection_boxes:0')
        self.scores = graph.get_tensor_by_name('detection_scores:0')
        self.classes = graph.get_tensor_by_name('detection_classes:0')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.tf_session = tf.Session(graph=graph, config=config)

        self.light_state = TrafficLight.UNKNOWN
        self.light_state_dict = {0: TrafficLight.GREEN,
                                 1: TrafficLight.RED,
                                 2: TrafficLight.YELLOW,
                                 3: TrafficLight.UNKNOWN}

    def load_graph(self, graph_path):
        graph = tf.Graph()
        with tf.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_in = np.copy(np.expand_dims(cv2.resize(image, (150, 200)), axis=0))
        (boxes, scores, classes) = self.tf_session.run([self.boxes, self.scores, self.classes], feed_dict={self.image_tensor: image_in})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        idx = classes[0]  # classes ==> 1-Green; 2-Red; 3-Yellow; 4-Unknown
        self.light_state = self.light_state_dict[idx-1]

        return self.light_state
