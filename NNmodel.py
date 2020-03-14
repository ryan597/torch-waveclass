from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
# Silence the tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Fix the random seed
tf.random.set_seed(2020)
################################################################################

#               TO-DO:
# improve generator - needs to be used for large data stream
# test architectures ()
# implement other CNN models
# adaptive learning rate callback                                           DONE
# streamline pipeline
# generalise code for paths (raw and flow)
# implement callbacks in fit (save model if validation doesn't increase
# check the precision and recall, ...)
# fine tuneing
# compare to Buscombes on the new split
# clean up and refactor

################################################################################

# Create the data pipeline to load images and labels

def get_image_generators(IMG_SIZE, 
                         BATCH_SIZE,
                         VAL_BATCH_SIZE
                         ):
    print("\ncreating image generators...")
    train_datagen = ImageDataGenerator(samplewise_center=True,
                                       samplewise_std_normalization=True,
                                       rotation_range=10,
                                       zoom_range=[0.8, 1.2],
                                      )
 
    print("fetching images for training...")
    train_data = train_datagen.flow_from_directory("train",
                                        target_size=(IMG_SIZE, IMG_SIZE),
                                        color_mode="rgb",
                                        batch_size=BATCH_SIZE,
                                        class_mode="categorical",
                                        #classes=['nonbreaking', 'plunge', 'spill'],
                                        shuffle=True,
                                        seed=2018)

    print("\nfetching images for validation...")
    valid_datagen = ImageDataGenerator(samplewise_center=True,
                                       samplewise_std_normalization=True
                                       )
    valid_data = valid_datagen.flow_from_directory("valid",
                                        target_size=(IMG_SIZE, IMG_SIZE),
                                        color_mode="rgb",
                                        batch_size=VAL_BATCH_SIZE,
                                        class_mode="categorical",
                                        #classes=['nonbreaking', 'plunge', 'spill'],
                                        shuffle=False,
                                        seed=2018)

    print("\nfetching images for testing...")
    test_datagen = ImageDataGenerator(samplewise_center=True,
                                      samplewise_std_normalization=True
                                      )
    test_data = test_datagen.flow_from_directory("test",
                                        target_size=(IMG_SIZE, IMG_SIZE),
                                        color_mode="rgb",
                                        batch_size=1,
                                        class_mode='categorical',
                                        #classes=['nonbreaking', 'plunge', 'spill'],
                                        shuffle=False,
                                        seed=2018)

    return train_data, valid_data, test_data

def lr_schedule(epoch):
    if epoch < 4:
        return 0.001
    else:
        return 0.001 * np.exp(0.2* (3 - epoch))

def build_model():
    optimizer = 'Adam'#get_optimizer()

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top = False,
                                                    weights='imagenet',
                                                    pooling='max')

    #base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
    #                                                include_top=False,
    #                                                weights='imagenet',
    #                                                pooling='max')

    # Freeze base_model (the CNN)
    base_model.trainable = False

    model = keras.Sequential([
        keras.layers.BatchNormalization(input_shape=IMG_SHAPE),
        base_model,
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),
        #keras.layers.Dense(16, activation='relu'),
        #keras.layers.Dropout(0.5),
        #keras.layers.BatchNormalization(),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    model.summary()
    return model

def fit_model(model, 
              train_data, 
              valid_data, 
              STEP_SIZE_TRAIN, 
              STEP_SIZE_VALID, 
              initial_epochs
              ):
    
    class_weights = class_weight.compute_class_weight('balanced',
                                    np.unique(train_data.classes),
                                    train_data.classes)

    #lrcallback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

    filepath = "SAVED_MODELS/model_{epoch:02d}_{val_accuracy:.2f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy',
                verbose=1, save_best_only=True, mode='max')

    stopcallback = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', min_delta=0, patience=4, verbose=1, mode='auto',
                baseline=None, restore_best_weights=True
                )

    lrplateu = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.1, patience=2, verbose=1, mode='auto',
                min_delta=0.0001, cooldown=0, min_lr=0
                )


    history = model.fit(
                        train_data,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_data,
                        validation_steps=STEP_SIZE_VALID, 
                        class_weight=class_weights,
                        epochs=initial_epochs,
                        callbacks=[stopcallback, lrplateu, checkpoint],
                        verbose=1)

    return history

def save_model(model, history):
    model_json = model.to_json()
    timestamp = time.strftime("%m%d-%H%M")
    valacc = int(history.history['val_accuracy'][-1] * 100)
    with open(f'SAVED_MODELS/model_{timestamp}_{valacc}.json', 'w') as json_file:
       json_file.write(model_json)
    model.save_weights(f'SAVED_MODELS/model_{timestamp}_{valacc}.h5')
    print(f"saved model and weights: \tmodel_{timestamp}_{valacc}")

def load_model(model):
    json_file = open(f'SAVED_MODELS/{model}.json', 'r')
    loaded_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights(f'SAVED_MODELS/{model}.h5')
    print(f"loaded model and weights")
    loaded_model.summary()

    return loaded_model

def plot_history(history):
    acc = history.history['accuracy']
    valacc = history.history['val_accuracy']
    loss = history.history['loss']
    valloss = history.history['val_loss']

    plt.rcParams.update({'font.size':10})
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(valacc, label="Valadation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(loss, label="Training Loss")
    plt.plot(valloss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

def show_batch(image_batch, 
               label_batch
               ):
  plt.rcParams.update({'font.size':8})
  plt.figure(figsize=(10,10))
  for n in range(25):
      plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      #plt.title(CLASS_NAMES[np.argmax([label_batch[n] == 1])])
      plt.axis('off')
      plt.pause(0.1)
  plt.show()
  
def validation_report(model, valid_data, VAL_BATCH_SIZE, STEP_SIZE_VALID):
    print("\ngenerating validation report...")
    p = model.predict(valid_data, steps=STEP_SIZE_VALID)
    preds = np.zeros_like(p)
    preds[np.arange(len(p)), p.argmax(1)] = 1
    labels = np.zeros_like(preds)
    for i in range(STEP_SIZE_VALID):
        imgBatch, labelBatch = next(valid_data)
        labels[i*VAL_BATCH_SIZE: (i+1)*VAL_BATCH_SIZE] = labelBatch
    print(classification_report(labels, preds))

####################################################

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 50
VAL_BATCH_SIZE = 10
initial_epochs=20

# Setting up the datasets
train_data, valid_data, test_data = get_image_generators(IMG_SIZE, BATCH_SIZE, VAL_BATCH_SIZE)

STEP_SIZE_TRAIN=train_data.n//train_data.batch_size
STEP_SIZE_VALID=valid_data.n//valid_data.batch_size


# Visualisation of sample images
#for img_batch, lab_batch in next(train_data):
#    show_batch(img_batch, lab_batch)
#####################################################

# Train the NN model
model = build_model()
validation_report(model, valid_data, VAL_BATCH_SIZE, STEP_SIZE_VALID)

history = fit_model(model, train_data, valid_data, STEP_SIZE_TRAIN, STEP_SIZE_VALID, initial_epochs)

validation_report(model, valid_data, VAL_BATCH_SIZE, STEP_SIZE_VALID)

plot_history(history)
# save the model and weights
save_model(model, history)

#####################################################

# Fine tuning of model
fine_epochs = 10
total_epochs = initial_epochs + fine_epochs


#history_fine = model.fit(train_data,
#                        epochs=total_epochs,
#                        initial_epoch=history.epoch[-1],
#                        validation_data=valid_balance,)
"""
history += history_fine
plot(history)


"""