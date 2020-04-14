FROM tensorflow/tensorflow:1.15.2-py3

ARG FLASK_ENV=production
ARG SEGMENTATION_MODEL_URL=https://storage.googleapis.com/ppl-eraser-static-bucket/api/segmentation_model.hdf5
ADD ${SEGMENTATION_MODEL_URL} /app/ppl_eraser_api/static/
COPY requirements.txt /app/
COPY ppl_eraser_api /app/ppl_eraser_api
COPY cmd.sh /app/
WORKDIR /app/

RUN apt-get update -y && \
    apt-get install -y libsm6 libxext6 libxrender1 libfontconfig1 git && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install gunicorn

#  tensorflow cannot load 'tensorflow_core.keras' with flask debug on
ENV FLASK_DEBUG=0
ENV FLASK_ENV=${FLASK_ENV}
ENV FLASK_APP="ppl_eraser_api:create_app('${FLASK_ENV}')"
ENV FLASK_RUN_PORT=50505
EXPOSE ${FLASK_RUN_PORT}

RUN groupadd -r ppl_eraser && useradd -r -g ppl_eraser ppl_eraser
RUN chown -R ppl_eraser:ppl_eraser /app
USER ppl_eraser

CMD ["./cmd.sh"]
