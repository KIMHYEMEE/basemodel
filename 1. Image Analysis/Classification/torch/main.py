import os
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LambdaCallback
from sklearn.metrics import confusion_matrix
import numpy as np

args = {'height': 200,
        'width': 200,
        'ch': 3,
        'output_size':2}

set_dir = 'C:/BaseData/Project'
model_dir = 'C:/BaseData/Project/MODEL'

os.chdir(model_dir)
import modeling

os.chdir(set_dir)


# 0. Functions ###############################################################

def get_pred(model, data):
    pred = model.predict_generator(data)
    pred_y = []
    for i in range(np.shape(pred)[0]):
        pred_y.append(np.argmax(pred[i]))
    pred = np.asarray(pred_y)
    pred = np.expand_dims(pred, axis=1)

    return pred


# 1. load data set ###########################################################
### data generation

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   zoom_range=0.3,
                                   validation_split=0.3,
                                   featurewise_center=True)
train_generator = train_datagen.flow_from_directory(
    './TRAIN',
    target_size=(args['height'], args['width']),
    batch_size=30,
    class_mode='categorical')

train_label = train_generator.classes[train_generator.index_array]

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  featurewise_center=True)
test_generator = test_datagen.flow_from_directory(
    './TEST',
    shuffle=False,
    target_size=(args['height'], args['width']),
    batch_size=30,
    class_mode='categorical')

test_label = test_generator.classes[test_generator.index_array]

# 2. modeling ################################################################
model = modeling.ResNet101(args)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])

# 3. Training ################################################################
print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))
early_stopping = EarlyStopping(patience=15, mode='auto', monitor='loss')
history = model.fit_generator(train_generator,
                              # steps_per_epoch=25,
                              epochs=100,
                              # callbacks=[early_stopping, print_weights]
                              callbacks=[early_stopping]
                              )

# 4. Performance #############################################################
train_generator = train_datagen.flow_from_directory(
    './TRAIN',
    target_size=(args['height'], args['width']),
    batch_size=30,
    shuffle=False,
    class_mode='categorical')

train_generator.reset()
test_generator.reset()

print("-- Evaluate(train) --")
scores = model.evaluate_generator(train_generator)
pred = get_pred(model, train_generator)

print(confusion_matrix(np.transpose(train_label), pred))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("-- Evaluate(test) --")
scores = model.evaluate_generator(test_generator)
pred = get_pred(model, test_generator)

print(confusion_matrix(np.transpose(test_label), pred))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))