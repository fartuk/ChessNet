from model import get_model
from loader import keras_generator
import keras

best_w = keras.callbacks.ModelCheckpoint('weights/13_01_best.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

last_w = keras.callbacks.ModelCheckpoint('weights/13_01_last.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=False,
                                save_weights_only=True,
                                mode='auto',
                                period=1)


callbacks = [best_w, last_w]

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = get_model()
model.compile(adam, 'binary_crossentropy')

batch_size = 16
model.fit_generator(keras_generator(),
              steps_per_epoch=20000,
              epochs=100,
              verbose=1,
              callbacks=callbacks,
              validation_data=keras_generator(),
              validation_steps=2000,
              class_weight=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              shuffle=True,
              initial_epoch=0)
