#!/bin/sh

export BUCKET=$1
export PROJECT_ID=$2

gsutil mb -c regional -l europe-west3 gs://${BUCKET}

#Upload all three component specifications to your Google Cloud Storage and make it public accessible by setting the permission to allUsers.
echo "\nCopy component specifications to Google Cloud Storage"
#gsutil cp preprocess/component.yaml gs://${BUCKET}/components/preprocess/component.yaml
#gsutil acl ch -u AllUsers:R gs://${BUCKET}/components/preprocess/component.yaml

gsutil rm gs://${BUCKET}/components/train/component.yaml
gsutil cp train/component.yaml gs://${BUCKET}/components/train/component.yaml
gsutil acl ch -u AllUsers:R gs://${BUCKET}/components/train/component.yaml

gsutil rm gs://${BUCKET}/components/deploy/component.yaml
gsutil cp deploy/component.yaml gs://${BUCKET}/components/deploy/component.yaml
gsutil acl ch -u AllUsers:R gs://${BUCKET}/components/deploy/component.yaml
