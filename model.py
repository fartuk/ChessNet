from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D
import keras

def get_model():
    inp = Input(shape=(8, 8, 12))
    conv_1 = Conv2D(512, (3,3), padding='same')(inp)
    conv_1 = Conv2D(512, (3,3), padding='same')(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(256, (3,3), padding='same')(conv_1)
    conv_2 = Conv2D(256, (3,3), padding='same')(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_3 = Conv2D(128, (3,3), padding='same')(conv_2)
    conv_3 = Conv2D(128, (3,3), padding='same')(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_4 = Conv2D(81, (3,3), padding='same')(conv_3)
    conv_4 = Conv2D(81, (3,3), padding='same')(conv_4)
    result = Activation('sigmoid')(conv_4)

    model = Model(inputs=inp, outputs=result)
    
    return model