from keras.models import model_from_json
import os

PATH = '.\\saved_models'


def save_model(model, name):
    label_model = model.to_json()
    with open(os.path.join(PATH, '{name}.json'), "w") as json_file:
        json_file.write(label_model)
    model.save_weights(os.path.join(PATH, f'{name}_weights.h5'))


def load_model(name):
    json_file = open(os.path.join(PATH, f'{name}.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(PATH, f'{name}_weights.h5'))

    return loaded_model
