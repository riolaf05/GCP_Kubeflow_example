#!/bin/sh

BUCKET='ai-vqc-kubeflow-components'

python3 setup.py sdist --dist-dir=.
gsutil cp custom_prediction_routine-0.2.tar.gz gs://${BUCKET}/routine/custom_prediction_routine-0.2.tar.gz
