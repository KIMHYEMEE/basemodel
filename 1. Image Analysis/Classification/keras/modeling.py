from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization

def ResNet101(args):
    from keras.applications import ResNet101

    input = Input(shape=(args['height'], args['width'], args['ch']))
    model = ResNet101(input_tensor=input, include_top=False, weights=None, pooling='max')  # param

    x = model.output
    x = Dense(1024, name='fully', kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(512, kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(args['output_size'], activation='softmax', name='softmax')(x)
    model = Model(model.input, x)

    return model