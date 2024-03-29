import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Input
from config import CFG
from datagenerator import data_generator_3D
from loss import loss
from model import UNet, UNet_RCAN, RCAN
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
print(len(gpus), "Physical GPUs")

data_config = CFG['data']
model_config = CFG['model']
callback = CFG['callbacks']
x_train, y_train = data_generator_3D(data_config)

mip_x_train = np.max(x_train, axis=3)
mip_y_train = np.max(y_train, axis=3)

ix = np.random.randint(0, len(x_train), 4)
fig = plt.figure(figsize=(15, 7))

for i in range(4):
    fig.add_subplot(2, 4, 2 * i + 1)
    cmap = plt.get_cmap('magma')
    plt.imshow(mip_x_train[ix[i], :, :, 0].squeeze(), cmap)
    plt.title('Low SNR', fontdict={'fontsize': 18})
    plt_axis = plt.axis('off')

    fig.add_subplot(2, 4, 2 * i + 2)
    cmap = plt.get_cmap('magma')
    plt.imshow(mip_y_train[ix[i], :, :, 0].squeeze(), cmap)
    plt.title('High SNR', fontdict={'fontsize': 18})
    plt_axis = plt.axis('off')
plt.show()

model_input = Input(
    (data_config['patch_size'], data_config['patch_size'], data_config['fr_end'] - data_config['fr_start'], 1))
model = eval(model_config['model_type'] + "(model_input, model_config)")
print(model.summary())

optimizer = keras.optimizers.Adam(learning_rate=model_config['lr'])
model.compile(optimizer=optimizer, loss=loss)

callbacks = [
    EarlyStopping(patience=callback['patience_stop'], verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=callback['factor_lr'], patience=callback['patience_lr']),
    ModelCheckpoint(filepath=model_config['save_dr'], verbose=1, save_best_only=True, save_weights_only=True)]

with open(os.path.join(model_config['save_config'], '', 'configuration.txt'), 'w') as data:
    data.write(str(CFG['model']))

results = model.fit(x=x_train, y=y_train, batch_size=model_config['batch_size'],
                    epochs=model_config['n_epochs'],
                    verbose=1, callbacks=callbacks, validation_split=0.1)
