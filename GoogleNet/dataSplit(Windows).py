import os
import shutil
fromName='6disease_new_origin'
toName='6disease_new'
classes=os.listdir('./'+fromName)
names=classes
for name in names:
    if not os.path.exists("./"+toName+"/train/"+name):
        os.makedirs("./"+toName+"/train/"+name)
        os.makedirs("./"+toName+"/test/"+name)
rate=0.8
ii=0
for classe in classes:
    tempDic=os.listdir('./'+fromName+'/'+classe)
    tempLenth=int(0.8*len(tempDic))
    src = './'+fromName+'/'+classe
    dist1='./'+toName+'/train/'+classe
    dist2 = './'+toName+'/test/' + classe
    ii+=1
    for i in range(len(tempDic)):
        if i<tempLenth:
            shutil.copy(src+'/'+tempDic[i],dist1+'/'+tempDic[i])
        else:
            shutil.copy(src + '/' + tempDic[i], dist2 + '/' + tempDic[i])