import os

import tensorflow as tf
from flask import Flask, make_response, url_for

from .config import configs
from .inpaint.inpainter import Inpainter


def create_app(config_name: str = 'development', config_dict: dict = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    if config_dict:
        app.config.from_mapping(config_dict)
    elif config_name:
        app.config.from_object(configs[config_name])
        configs[config_name].init_app(app)

    from .routes import bp
    app.register_blueprint(bp, url_prefix='/api')

    with app.app_context():
        g = tf.Graph()
        with g.as_default():
            from .custom_model_objects import bce_dc_loss, mean_iou
            sess = tf.compat.v1.Session(graph=g)
            tf.compat.v1.keras.backend.set_session(sess)
            segmentation_model = tf.compat.v1.keras.models.load_model(
                app.config['SEGMENTATION_MODEL_PATH'],
                custom_objects={
                    'mean_iou': mean_iou,
                    'bce_dc_loss': bce_dc_loss,
                },
            )
            segmentation_model._make_predict_function()
        app.config['segmentation_model_graph'] = g
        app.config['segmentation_model_session'] = sess
        app.config['segmentation_model'] = segmentation_model

        inpainter = Inpainter(
            app.config['INPAINTER_CHECKPOINT_DIR'],
            app.config['INPAINTER_NG_CONFIG_PATH'],
        )
        app.config['inpainter'] = inpainter

    @app.route('/readiness_check', methods=['GET'])
    def readiness_check():
        """ Enpoint for App Engine to check server readiness. """
        return 'OK', 200

    # No cacheing at all for API endpoints.
    @app.after_request
    def add_header(response):
        """ After request response modification to set no caching. """
        if 'Cache-Control' not in response.headers:
            response.headers['Cache-Control'] = 'no-store'
        return response

    return app
