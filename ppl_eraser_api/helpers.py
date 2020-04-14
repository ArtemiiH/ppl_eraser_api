import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize


def mask_image(image, mask):
    img_masked = resize(image.copy(), mask.shape[:2])
    img_masked[mask > 0] = 1.0
    return img_masked


def get_predicted_rgb_mask_arr(model, image):
    image = image.astype(np.float32) / 255
    image = resize(image, (image.shape[0]//32 * 32, image.shape[1]//32 * 32))
    # expand dimensions to make it one image batch
    image = np.expand_dims(image, axis=0)[:, :, :, :3]
    # preds = model.predict(image).round()[0]
    preds = (model.predict(image) > 0.25)[0]
    # repeating preds 3 time so that it will be valid RGB image
    mask_arr = np.repeat(preds.astype(np.float32), repeats=3, axis=-1)
    return mask_arr


def load_image_from_b64string(string):
    decoded = BytesIO(base64.b64decode(string))
    image = Image.open(decoded)
    # imageIO = BytesIO()
    # image.save(imageIO, "PNG")
    # imageIO.seek(0)
    # image_arr = np.fromstring(imageIO.read(), dtype=np.uint8)
    # image_arr = image_arr.reshape((image.height, image.width, -1))
    image_arr = np.array(image, dtype=np.uint8)
    image_arr = image_arr[:, :, :3]
    return image_arr
