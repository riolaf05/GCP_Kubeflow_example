# GCP_Kubeflow_example
Example of generic deep learning application deployed on Kubeflow on GCP.

In this case study an italian **COVID-19** open dataset is used to perform prediction on virus spread. Also a **Kubeflow** pipeline will be created to host the predictions and monitor the results.

### Kubeflow Cluster creation (for DevOps)

See [here](https://codelabs.developers.google.com/codelabs/cloud-kubeflow-pipelines-gis/index.html?index=../..index#0)

1. Connect to: [Kubeflow deploy tool](https://deploy.kubeflow.cloud/)

2. Create Kubeflow cluster noting `PROJECT_ID`, `ZONE` and `DEPLOYMENT_NAME`. This tool will create Kubernetes deployment, services, ingress, etc.

3. If it works from local, launch `install_gcloud.sh` script to setup Google Cloud CLI, otherwise open `Gcloud Shell`.

4. Create a new bucket, install `kfp` and enable services:

```console
export DEPLOYMENT_NAME=kf-test
export PROJECT_ID=<PROJECT_NAME>
export ZONE=europe-west1-d
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

export BUCKET_NAME=${PROJECT_ID}-kubeflow
gsutil mb gs://${BUCKET_NAME}

sudo pip3 install -U kfp

gcloud services enable cloudresourcemanager.googleapis.com iam.googleapis.com file.googleapis.com ml.googleapis.com
```

5. Connect to Kubeflow cluster:

```console
gcloud container clusters get-credentials ${DEPLOYMENT_NAME} --project ${PROJECT_ID} --zone ${ZONE}
kubectl config set-context $(kubectl config current-context) --namespace=kubeflow
```

6. If not already present, create GPU node-pool:

```console
kubectl create clusterrolebinding sa-admin --clusterrole=cluster-admin --serviceaccount=kubeflow:pipeline-runner

gcloud container node-pools create gpu-pool \
    --cluster=${DEPLOYMENT_NAME} \
    --zone ${ZONE} \
    --num-nodes=1 \
    --machine-type n1-highmem-8 \
    --scopes cloud-platform --verbosity error \
    --accelerator=type=nvidia-tesla-k80,count=1
```

### Kubeflow pipeline creation (for Developers)

1. Authenticate into Google Cloud Docker Registry: `gcloud auth configure-docker`.

2. For each  component (train, deploy, etc.) use the `build_image.sh` scripts to build components:

```console
chmod +x ./build_image.sh.sh
./build_image.sh.sh 
```

this will build (reusable) and push components containers.

To build components directly it is possible to use the `dsl.ContainerOp` object (see [here](https://www.kubeflow.org/docs/pipelines/sdk/build-component/)) this is easier but those components are not reusable.  

See examples [here](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html).

3. For the deploy step run the `routine/build_routine.sh` script to build custom model execution script: 

```console
chmod +x ./routine/build_routine.sh
```

4. Use the Jupyter notebook `pipeline_creation.ipynb` into Kubeflow environment to build pipeline with given components.

NOTE: there is an issue with the "component upload" step of the `pipeline_creation.ipynb` workflow: sometimes updates on components YAML definition require time to be effective. This is due to some kind of cache in Kubeflow or GCS storage. 

### Invoke model on AI Platform

Use the script in `manual_build.ipynb` to invoke the endpoints (with bash or Python). 

Note: it is possible to depoy tensorflow, scikit-learn, or xgboost models. For any other custom prediction use "Custom prediction Routines" (see [docs](https://cloud.google.com/ai-platform/prediction/docs/custom-prediction-routine-keras#create_a_custom_predictor))

### Next steps: 

* Fix problem with unloaded metrics and tensorboard (see [Kubeflow output docs](https://www.kubeflow.org/docs/pipelines/sdk/output-viewer/#writing-out-metadata-for-the-output-viewers))
* Add validation step
* Add CI/CD pipeline to build Kubeflow components

### References:

[Kubeflow deploy example](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-1-setup.md)

[Introduction to AI Platform](https://cloud.google.com/ai-platform/docs/technical-overview?authuser=2)

[Deploying models](https://cloud.google.com/ai-platform/prediction/docs/deploying-models?authuser=2)

[Deploy Keras on Kubeflow](https://medium.com/@vincentweimer1/deploy-keras-model-on-gcp-and-making-custom-predictions-via-the-ai-platform-training-prediction-16e0213470d4)

[Kubeflow Pipeline Docs](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)

