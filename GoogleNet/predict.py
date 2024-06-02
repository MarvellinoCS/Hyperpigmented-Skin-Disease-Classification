import traceback
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from Gnet import GoogLeNet
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'font.size': 10})
save_model_name='GoogLeNet6.h5'
totalName='4disease_new'
class_num=4
names=os.listdir('./'+totalName+'/test')
def imagesModel():
    model = GoogLeNet(class_num=class_num, aux_logits=False)
    model.summary()
    model.load_weights("./save_weights/"+save_model_name, by_name=True)
    im_height = 224
    im_width = 224
    imageFile=names
    imagesModelRes=[]
    count=0
    nonLabel=[]
    predicts=0
    trueLabel=[]
    abilitiable=[]
    for imagePath in imageFile:
        for dir in os.listdir('./'+totalName+'/test/'+str(imagePath)):
            img = Image.open('./'+totalName+'/test/'+str(imagePath)+'/'+str(dir)).convert('RGB')
            img = img.resize((im_width, im_height))
            img = ((np.array(img) / 255.) - 0.5) / 0.5
            #print(len(img),len(img[0]),len(img[0][0]))
            img = (np.expand_dims(img, 0))
            try:
                result = model.predict(img)
                trueLabel.append(names.index(imagePath)+1)
                abilitiable.append(result)
                if np.argmax(result)+1 == names.index(imagePath)+1:
                    predicts+=1
                imagesModelRes.append(np.argmax(result)+1)
            except Exception as e:
                print('str(e):\t\t', str(e))
                print('repr(e):\t', repr(e))
                nonLabel.append(count)
                print('First',count,'Exception！')
                print('./'+totalName+'/test/'+str(imagePath)+'/'+str(dir))
            count+=1
    return imagesModelRes,nonLabel,float(predicts/count),trueLabel,abilitiable
def auc1(trueLabel,abiliable,classes=class_num):
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
            tempAbiliable[j]=abiliable[j][0][i-1]
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
    save_name=['./GoogLeNet-Algorithm-to-0-3-class-disease-sizeClass.png']
    lens=1
    for temp_save_name in save_name:
        for i in range(class_num):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,label='ROC curve of class{0} (AUC area = {1:0.2f})'.format(str(i), roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Sensitivity')
        plt.ylabel('Specificity')
        plt.title('GoogLeNet')
        plt.legend(loc="lower right")
        plt.savefig(temp_save_name, format='png')
        plt.clf()
        lens+=1
    #plt.show()
def matrixPlot(imagesModelRes,trueLabel):
    cm=confusion_matrix(trueLabel, imagesModelRes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title('GoogLeNet')
    plt.colorbar()
    labels_name=[ str(i) for i in range(class_num)]
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./confusion_matrix.png', format='png')
    plt.show()
def saveParamIndex(trueLabel,predictLabel,class_num=class_num):
    result={}
    for i in range(class_num):
        TP=0
        TN=0
        FP=0
        FN=0
        for j in range(len(predictLabel)):
            if predictLabel[j]==trueLabel[j] and trueLabel[j]==i:
                TP+=1
            elif predictLabel[j]==trueLabel[j] and trueLabel[j]!=i:
                TN+=1
            elif predictLabel[j]!=trueLabel[j] and trueLabel[j]==i:
                FN+=1
            elif predictLabel[j]!=trueLabel[j] and trueLabel[j]!=i:
                FP+=1
        Accuracy=(TP+TN)/float(TP+TN+FP+FN)
        Sensitivity=TP / float(TP+FN)
        Specificity=TN / float(TN+FP)
        result[names[i]]=[Accuracy,Sensitivity,Specificity]
    return result
if __name__ == '__main__':
    imagesModelRes,nonLabel,accuracy,trueLabel,abiliable=imagesModel()
    print('accuracy:', accuracy)
    matrixPlot(imagesModelRes, trueLabel)
    fpr, tpr, roc_auc = auc1(trueLabel, abiliable)
    plotPictrue(fpr, tpr, roc_auc)
    result = saveParamIndex(trueLabel, imagesModelRes)
    print(result)
