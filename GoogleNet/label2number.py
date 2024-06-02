import os

origin_file_name = '6disease_new_origin'

def main():
    classes = os.listdir('./' + origin_file_name)
    label2number = open(origin_file_name + '_label2number.csv', 'w', encoding='gbk')
    info = 'index,class\n'  # Initialize info
    for index, cla in enumerate(classes):
        print(index, cla)
        templine = str(index) + ',' + str(cla) + '\n'
        info += templine
    label2number.write(info)
    label2number.close()
    return classes

def rename_dir(classes):
    print(os.listdir('./' + origin_file_name))  # Correct the directory path
    fix_name = origin_file_name + '/'
    for index, cla in enumerate(classes):
        src, dist = fix_name + str(cla), fix_name + str(index)
        os.rename(src, dist)
    print(os.listdir('./' + origin_file_name))  # Correct the directory path

if __name__ == '__main__':
    classes = main()
    rename_dir(classes)
