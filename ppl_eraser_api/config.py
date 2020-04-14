import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'DobryTajemyKlucz01!!'

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = 0
    SEGMENTATION_MODEL_PATH = os.environ.get('DEV_SEGMENTATION_MODEL_PATH') \
        or "./ppl_eraser_api/static/segmentation_model.hdf5"
    INPAINTER_CHECKPOINT_DIR = os.environ.get('DEV_INPAINTER_CHECKPOINT_DIR') \
        or "./ppl_eraser_api/static/inpaint/checkpoint"
    INPAINTER_NG_CONFIG_PATH = os.environ.get('DEV_INPAINTER_NG_CONFIG_PATH') \
        or "./ppl_eraser_api/static/inpaint/inpaint.yml"

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)


class ProductionConfig(Config):
    DEBUG = 0
    SEGMENTATION_MODEL_PATH = os.environ.get('SEGMENTATION_MODEL_PATH') \
        or "./ppl_eraser_api/static/segmentation_model.hdf5"
    INPAINTER_CHECKPOINT_DIR = os.environ.get('INPAINTER_CHECKPOINT_DIR') \
        or "./ppl_eraser_api/static/inpaint/checkpoint"
    INPAINTER_NG_CONFIG_PATH = os.environ.get('INPAINTER_NG_CONFIG_PATH') \
        or "./ppl_eraser_api/static/inpaint/inpaint.yml"

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)


configs = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
