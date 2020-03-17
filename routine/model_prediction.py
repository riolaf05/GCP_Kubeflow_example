import os
import pickle
import numpy as np
import tensorflow as tf

class CustomModelPrediction():

  def __init__(self, model):
    self._model = model
    #self._processor = processor
  
  '''
  def postprocess(self, predictions):
    labeled_predictions = []

    for prediction in predictions:
      labeled_prediction = []
      for word_prediction in prediction:
        labeled_prediction.append(self._processor.labels[word_prediction])
      labeled_predictions.append(labeled_prediction)

    return labeled_predictions
  '''
  def predict(self, instances):
    #transformed_instances = self._processor.transform(instances)
    #predictions = self._model.predict(np.array(transformed_instances))
    predictions = self._model.predict(np.array(instances))
    predictions = np.argmax(predictions, axis=-1).tolist()

    #labels = self.postprocess(predictions)
    #return labels
    return predictions

  @classmethod
  def from_path(cls, model_dir): #NOTE: commenting pre-processsor code due to disuse
    model = tf.keras.models.load_model(os.path.join(model_dir, 'keras_saved_model.h5'))
    #with open(os.path.join(model_dir, 'preprocessor.pkl'), 'rb') as f:
    #  processor = pickle.load(f)

    return cls(model)
