import os
import pandas
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline
from keras.utils import to_categorical
import argparse
from pathlib import Path
from google.cloud import storage
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import json
import numpy as np

PROJECT='ai-vqc'
INPUT_BUCKET='ai-vqc-kubeflow-test'
OUTPUT_BUCKET='ai-vqc-kubeflow-output'
PREFIX=''
INPUT_PATH=r'/opt/ml/input/'
#INPUT_PATH=r'/home/rosario/Codice/kubeflow-test/train/data/'
OUTPUT_PATH='/opt/ml/model'
MODEL_FILE = 'keras_saved_model.h5'

def read_from_gcs(filename, path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(INPUT_BUCKET)
    blob = bucket.blob(PREFIX+filename)
    blob.download_to_filename(path+filename)
    print("Data downloaded to ", path+filename)

def write_to_gcs(filename, path):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(OUTPUT_BUCKET)
    blob = bucket.blob(filename)
    blob.upload_from_filename(path+filename)
    print(path+filename, " Model written to ", OUTPUT_BUCKET, " bucket.")

def save_model(model, output_path): #Save model file to GCS
    print('saved model to ', output_path)
    model.save(MODEL_FILE)
    with file_io.FileIO(MODEL_FILE, mode='rb') as input_f:
        with file_io.FileIO(output_path + '/' + MODEL_FILE, mode='wb+') as output_f:
            output_f.write(input_f.read())

def copy_local_directory_to_gcs(local_path, bucket_name, gcs_path): #copy on GSP with SavedModel format
    """Recursively copy a directory of files to GCS.

    local_path should be a directory and not have a trailing slash.
    """
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            continue
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        remote_path = os.path.join(gcs_path, local_file[1 + len(local_path)-1 :])
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)


def train(input_path, output):

    # initialize tensorboard
    tensorboard = TensorBoard(
    log_dir=os.path.join(input_path, 'logs'),
    histogram_freq=0,
    write_graph=True,
    embeddings_freq=0)
    callbacks = [tensorboard]

    # load dataset
    dataframe = pandas.read_csv(input_path)
    #dataset = dataframe.values
    #dataframe.head()

    X = dataframe.drop(columns=['totale_casi']).drop(columns=['data']).drop(columns=['stato']).drop(columns=['denominazione_regione'])
    Y = dataframe[['totale_casi']]
    Y_categorical=to_categorical(Y)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y_categorical, test_size=0.2)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=X_train.shape[1]))
    model.add(tf.keras.layers.Dense(256, activation='relu')) 
    model.add(tf.keras.layers.Dense(128, activation='relu')) 
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)

    loss, accuracy = model.evaluate(X_test, np.array(y_test))

    save_model(model, output) #To save Keras or custom models
    #saved_to_path = tf.contrib.saved_model.save_keras_model(model, output) #to save Tensorflow model

    # write out metrics (see: https://www.kubeflow.org/docs/pipelines/sdk/output-viewer/)
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue': str(accuracy),
            'format': "PERCENTAGE",
        }]
    }
    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        print(metrics)
        json.dump(metrics, f)

    # write out TensorBoard viewer
    metadata = {
        'outputs': [{
            'type': 'tensorboard',
            'source': 'gs://ai-vqc-ai-vqc-kubeflow-output/model',
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

def main():
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('--input-filename', type=str, help='Input filename.') # Paths should be passed in, not hardcoded
    parser.add_argument('--output-model-path', type=str, help='GCS output path') # Paths should be passed in, not hardcoded
    parser.add_argument('--output-model-path-file', type=str, help='Filename path the output model.') # Output variable
    args = parser.parse_args()

    print(args.input_filename)
    print(args.output_model_path)
    print(args.output_model_path_file)

    read_from_gcs(args.input_filename, INPUT_PATH)
    train(INPUT_PATH+args.input_filename, args.output_model_path)
    #copy_local_directory_to_gcs(OUTPUT_PATH+'/test/', args.output-model-path, 'test') #copy on SavedModel format

    #Export output model to output variable 
    Path(args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_path_file).write_text(args.output_model_path)

if __name__ == "__main__":
    main()