import os

import neuralgym as ng
import numpy as np
import tensorflow as tf
from skimage import io

from .inpaint_model import InpaintCAModel


class Inpainter:

    def __init__(self, checkpoint_dir: os.PathLike, ng_config_path: os.PathLike) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.FLAGS = ng.Config(ng_config_path)

    def prepare_model_input(self, image: np.ndarray[int], mask: np.ndarray[int]) -> np.ndarray[int]:
        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        return input_image

    def inpaint(self, image: np.ndarray[int], mask: np.ndarray[int]) -> np.ndarray[int]:
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        input_image = self.prepare_model_input(image, mask)
        # ng.get_gpus(1)
        model = InpaintCAModel()
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(self.FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                var_value = tf.contrib.framework.load_variable(
                    self.checkpoint_dir, var.name)
                assign_ops.append(tf.compat.v1.assign(var, var_value))
            sess.run(assign_ops)
            result = sess.run(output)
        return result[0][:, :, ::-1]
