import numpy as np
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from keras.initializers import glorot_uniform
longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = keras.models.load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    res="La mamografia predecida se clasifico en nivel 0 Birads"
  elif answer == 1:
    res="La mamografia predecida se clasifico en nivel 1 Birads"
  elif answer == 2:
    res="La mamografia predecida se clasifico en nivel 2 Birads"
  elif answer == 3:
    res="La mamografia predecida se clasifico en nivel 3 Birads"
  elif answer == 4:
    res="La mamografia predecida se clasifico en nivel 4 Birads"
  elif answer == 5:
    res="La mamografia predecida se clasifico en nivel 5 Birads"
  elif answer == 6:
    res="La mamografia predecida se clasifico en nivel 6 Birads" 
  print(res)
  
predict('./prueba.png')
