from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Gnet import GoogLeNet
import tensorflow as tf
import os
import time

start_time = time.perf_counter()
image_path = "4disease_new/"
train_dir = image_path + "train"
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")
im_height = 224
im_width = 224
batch_size = 32
epochs = 100
class_num = 4
save_model_name = 'GoogLeNet6.h5'


def pre_function(img):
    img = img / 255.
    img = (img - 0.5) * 2.0
    return img


train_image_generator = ImageDataGenerator(preprocessing_function=pre_function, horizontal_flip=True)
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, batch_size=batch_size, shuffle=True,
                                                           target_size=(im_height, im_width), class_mode='categorical')
total_train = train_data_gen.n
model = GoogLeNet(im_height=im_height, im_width=im_width, class_num=class_num, aux_logits=True)
model.summary()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(images, training=True)
        loss1 = loss_object(labels, aux1)
        loss2 = loss_object(labels, aux2)
        loss3 = loss_object(labels, output)
        loss = loss1 * 0.3 + loss2 * 0.3 + loss3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, output)


best_test_loss = float('inf')
best_acc = 0.0
for epoch in range(1, epochs + 1):
    train_loss.reset_states()
    train_accuracy.reset_states()
    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_step(images, labels)
    template = 'Epoch {}, training Loss: {}, training Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result() * 100))
    if train_loss.result() < best_test_loss:
        best_test_loss = train_loss.result()
        best_acc = train_accuracy.result()
        model.save_weights("./save_weights/" + save_model_name)
print('best_acc', best_acc)
print('totalRunTime', (time.perf_counter() - start_time) / 60.0, 'min')
