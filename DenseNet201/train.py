from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import time
start_time=time.perf_counter()
im_height = 224
im_width = 224
batch_size = 32
epochs = 50
classNum=4
image_path = "./4disease_new/"
train_dir = image_path + "train"
rename='DenseNet201'
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")
train_image_generator = ImageDataGenerator(rescale=1. / 255,  rotation_range=40, width_shift_range=0.2,  height_shift_range=0.2,  shear_range=0.2,  zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, batch_size=batch_size,shuffle=True,  target_size=(im_height, im_width), class_mode='categorical')
total_train = train_data_gen.n
covn_base = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(im_width, im_height, 3))
covn_base.trainable = True
model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(classNum, activation='softmax'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),metrics=["accuracy"])
history = model.fit(x=train_data_gen,steps_per_epoch=total_train // batch_size,epochs=epochs)
model.save_weights("./save_weights/"+rename+".h5")
print('totalRunTime',(time.perf_counter()-start_time)/60.0,'min')