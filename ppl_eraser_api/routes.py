import base64
import time
from io import BytesIO

import tensorflow as tf
from flask import (Blueprint, current_app, jsonify, make_response, request,
                   send_file)
from PIL import Image
from skimage.filters import gaussian

from ppl_eraser_api.helpers import (get_predicted_rgb_mask_arr,
                                    load_image_from_b64string, mask_image)

bp = Blueprint('api', __name__)


@bp.route('/mask', methods=['POST'])
def mask_endpoint():
    segmentation_model = current_app.config.get('segmentation_model')
    image = load_image_from_b64string(request.json.get('image'))
    with current_app.config.get('segmentation_model_graph').as_default():
        tf.compat.v1.keras.backend.set_session(
            current_app.config.get('segmentation_model_session'))
        mask_arr = get_predicted_rgb_mask_arr(segmentation_model, image)
    mask_arr *= 255
    # save to BytesIO object as png in order to send it back as file
    mask = Image.fromarray(mask_arr.astype('uint8'), 'RGB')
    mask_file = BytesIO()
    mask.save(mask_file, format='png')
    mask_file.seek(0)
    img_base64 = base64.b64encode(mask_file.read())
    resp = make_response(jsonify({'image': img_base64.decode('utf-8')}), 200)
    return resp


@bp.route('/cut', methods=['POST'])
def cut_endpoint():
    segmentation_model = current_app.config.get('segmentation_model')
    image = load_image_from_b64string(request.json.get('image'))
    with current_app.config.get('segmentation_model_graph').as_default():
        tf.compat.v1.keras.backend.set_session(
            current_app.config.get('segmentation_model_session'))
        mask = get_predicted_rgb_mask_arr(segmentation_model, image)
    masked_image = mask_image(image, mask) * 255
    # save to BytesIO object as png in order to send it back as file
    masked_image = Image.fromarray(masked_image.astype('uint8'), 'RGB')
    mask_file = BytesIO()
    masked_image.save(mask_file, format='png')
    mask_file.seek(0)
    img_base64 = base64.b64encode(mask_file.read())
    resp = make_response(jsonify({'image': img_base64.decode('utf-8')}), 200)
    return resp


@bp.route('/inpaint', methods=['POST'])
def inpaint_endpoint():
    segmentation_model = current_app.config.get('segmentation_model')
    image = load_image_from_b64string(request.json.get('image'))
    with current_app.config.get('segmentation_model_graph').as_default():
        tf.compat.v1.keras.backend.set_session(
            current_app.config.get('segmentation_model_session'))
        mask = get_predicted_rgb_mask_arr(segmentation_model, image)
    mask = gaussian(mask, sigma=1, multichannel=True)
    mask[mask > 0] = 1.0
    masked_image = mask_image(image, mask)
    # save to BytesIO object as png in order to send it back as file
    inpainter = current_app.config.get('inpainter')
    inpainted_arr = inpainter.inpaint(masked_image*255, mask*255)
    inpainted_image = Image.fromarray(inpainted_arr.astype('uint8'), 'RGB')
    inpainted_file = BytesIO()
    inpainted_image.save(inpainted_file, format='png')
    inpainted_file.seek(0)
    img_base64 = base64.b64encode(inpainted_file.read())
    resp = make_response(jsonify({'image': img_base64.decode('utf-8')}), 200)
    return resp
