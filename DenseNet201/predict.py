from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'font.size': 10})
im_height = 224
im_width = 224
batch_size = 64
classNum=4
names=os.listdir('./4disease_new/test/')
image_path = "./4disease_new/"
rename='DenseNet201'
validation_dir = image_path + "test"
validation_image_generator = ImageDataGenerator(rescale=1. / 255)
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir, batch_size=batch_size,shuffle=False,  target_size=(im_height, im_width),  class_mode='categorical')
total_val = val_data_gen.n
covn_base = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(im_width, im_height, 3))
model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(classNum, activation='softmax'))
model.summary()
model.load_weights('./save_weights/'+rename+'.h5')
def auc1(trueLabel,abiliable,classes=classNum):
    tempTrueLabel=[0]*len(trueLabel)
    tempAbiliable=[0]*len(trueLabel)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        i+=1
        for j in range(len(trueLabel)):
            if trueLabel[j]==i:
                tempTrueLabel[j]=1
            tempAbiliable[j]=abiliable[j][i-1]
        fpr[i-1], tpr[i-1], thresholds = roc_curve(tempTrueLabel, tempAbiliable, pos_label=1)
        roc_auc[i-1]=auc(fpr[i-1], tpr[i-1])
        tempTrueLabel = [0] * len(trueLabel)
        tempAbiliable = [0] * len(trueLabel)
    return fpr,tpr,roc_auc
def plotPictrue(fpr,tpr,roc_auc):
    lw = 2
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue','red','blue','green','black','bisque','burlywood','antiquewhite','tan','navajowhite',
     'goldenrod','gold','khaki','ivory','forestgreen','limegreen',
     'springgreen','lightcyan','teal','royalblue',
     'navy','slateblue','indigo','darkorchid','darkviolet','thistle']
    save_name=['./'+rename+'-Algorithm-to-0-3-class-disease-sizeClass.png']
    lens=1
    for temp_save_name in save_name:
        for i in range(classNum):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,label='ROC curve of class{0} (AUC area = {1:0.2f})'.format(str(i), roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Sensitivity')
        plt.ylabel('Specificity')
        plt.title(rename)
        plt.legend(loc="lower right")
        plt.savefig(temp_save_name, format='png')
        plt.clf()
        lens+=1
    #plt.show()
def matrixPlot(imagesModelRes,trueLabel):
    cm=confusion_matrix(trueLabel, imagesModelRes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(rename)
    plt.colorbar()
    labels_name=[ str(i) for i in range(classNum)]
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./confusion_matrix.png', format='png')
    plt.show()
def main():
    Y_pred = model.predict_generator(val_data_gen, total_val // batch_size+1)
    Y_pred_classes = np.argmax(Y_pred, axis=1)+np.array(1)
    accuracy=0
    trueLabel=val_data_gen.classes+np.array(1)
    for i in range(len(Y_pred_classes)):
        if Y_pred_classes[i]==trueLabel[i]:
            accuracy+=1
    print(accuracy/len(Y_pred_classes))
    matrixPlot(Y_pred_classes,val_data_gen.classes+np.array(1))
    fpr,tpr,roc_auc=auc1(val_data_gen.classes+np.array(1),Y_pred)
    plotPictrue(fpr, tpr, roc_auc)
if __name__ == '__main__':
    main()