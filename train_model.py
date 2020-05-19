from models import u_net, simple_autoencoder, SSIM
from pathlib import Path
from save_load_model import save_model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
import nrrd
from batch_generator import *
from keras.optimizers import Adam, SGD
import numpy as np

TRAIN_DATA_PATH = Path('.\\train_data\\noisy')
TRAIN_MASK_PATH = Path('.\\train_data\\ground_truth')
VAL_DATA_PATH = Path('.\\validation_data\\noisy')
VAL_MASK_PATH = Path('.\\validation_data\\ground_truth')
TEST_DATA_PATH = Path('.\\test_data\\noisy')
TEST_MASK_PATH = Path('.\\test_data\\ground_truth')

x_batch, y_batch = next(batch_generator(TRAIN_DATA_PATH, TRAIN_MASK_PATH, 4))

train_generator = batch_generator(TRAIN_DATA_PATH, TRAIN_MASK_PATH, 4)
validation_generator = batch_generator(VAL_DATA_PATH, VAL_MASK_PATH, 4)

# model = u_net(x_batch)
# model.summary()

callbacks = [
    ModelCheckpoint('best_model_denoising_ssim_100.h5', monitor='val_loss', mode='min', save_best_only=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
]

opt = Adam(learning_rate=1e-5)
opt2 = SGD(learning_rate=0.1e-5)

model = load_model('denoising_ssim_100')
model.compile(opt, loss=SSIM)

# trained_model = model.fit(train_generator, steps_per_epoch=2528, epochs=13, validation_data=validation_generator,
#                                  validation_steps=512, verbose=1, callbacks=callbacks)
# save_model(model, 'denoising_ssim_100')


for i in range(100):
    x_test, y_test = next(batch_generator(TEST_DATA_PATH, TEST_MASK_PATH, 16))
    y_hat = model.predict(x_test)
    nrrd.write('random_sample_X_{:03d}.nrrd'.format(i), np.squeeze(x_test), compression_level=1, index_order='C')
    nrrd.write('random_sample_Y_{:03d}.nrrd'.format(i), np.squeeze(y_test), compression_level=1, index_order='C')
    nrrd.write('random_sample_prediction_{:03d}.nrrd'.format(i), np.squeeze(y_hat), compression_level=1,
               index_order='C')
