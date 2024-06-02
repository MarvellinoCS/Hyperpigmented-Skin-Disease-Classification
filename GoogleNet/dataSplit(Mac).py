import os
import shutil

fromName = '6disease_new_origin'
toName = '6disease_new'
classes = os.listdir('./' + fromName)
names = classes
for name in names:
    if not os.path.exists("./" + toName + "/train/" + name):
        os.makedirs("./" + toName + "/train/" + name)
        os.makedirs("./" + toName + "/test/" + name)
rate = 0.8
ii = 0
for classe in classes:
    # Construct the directory path for the current class
    class_dir = './' + fromName + '/' + classe

    # Check if the path is a directory
    if not os.path.isdir(class_dir):
        # If not, skip to the next class
        continue

    tempDic = os.listdir(class_dir)
    tempLength = int(0.8 * len(tempDic))
    src = './' + fromName + '/' + classe
    dist1 = './' + toName + '/train/' + classe
    dist2 = './' + toName + '/test/' + classe
    ii += 1
    for i in range(len(tempDic)):
        if tempDic[i] == '.DS_Store':
            continue  # Skip .DS_Store files
        if i < tempLength:
            shutil.copy(src + '/' + tempDic[i], dist1 + '/' + tempDic[i])
        else:
            shutil.copy(src + '/' + tempDic[i], dist2 + '/' + tempDic[i])
