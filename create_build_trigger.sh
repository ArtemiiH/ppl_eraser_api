#!/bin/bash
gcloud beta builds triggers create cloud-source-repositories \
    --repo=ppl-eraser-api \
    --branch-pattern=".*" \
    --build-config=cloudbuild_app_engine.yaml
