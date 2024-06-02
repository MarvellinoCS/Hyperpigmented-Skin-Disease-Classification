from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from PIL import Image
import torchvision.transforms as transforms

# Configuration parameters
im_height = 224
im_width = 224
batch_size = 64
class_names = ['Cafe-au-lait Spot', 'Congenital-Nevus', 'Malignant-menanoma', 'Melasma']
class_name_to_index = {name: idx for idx, name in enumerate(class_names)}
validation_dir = "./4disease_new/test"
rename = 'YOLO'

# Load the YOLO model
model_path = './runs/classify/train/weights/last.pt'
model = YOLO(model_path)


def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((im_height, im_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def evaluate_model(model, validation_dir, class_name_to_index, conf_threshold=0.05):
    true_labels = []
    pred_labels = []
    all_preds = []


    for subdir, _, files in os.walk(validation_dir):
        true_class_name = os.path.basename(subdir)
        true_class = class_name_to_index.get(true_class_name, -1)
        if true_class == -1:
            continue

        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):

                img_path = os.path.join(subdir, file)

                # Get predictions from the YOLO model
                results = model.predict(img_path, conf=conf_threshold)

                # names_dict = results[0].names
                # proba = results[0].probs.data.tolist()
                # print(names_dict[np.argmax(proba)])
                # pred_name = names_dict[np.argmax(proba)]
                # print(pred_name)
                pred_class = results[0].probs.top1
                print(pred_class)

                if pred_class == true_class:
                    pred_class = true_class

                confidence = results[0].probs.top1conf

                true_labels.append(true_class)
                pred_labels.append(pred_class)
                pred_probs = [0] * len(class_names)
                if pred_class != -1:
                    pred_probs[pred_class] = confidence
                all_preds.append(pred_probs)

                print(f"Image: {file}")
                print(f"True Class: {true_class_name} ({true_class})")
                print(f"Predicted Class: {pred_class} with confidence {confidence}")

    print("True Labels:", true_labels)
    print("Predicted Labels:", pred_labels)

    return true_labels, pred_labels, all_preds



def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))

    if np.sum(cm) == 0:
        print("Confusion matrix is empty.")
        return

    row_sums = cm.sum(axis=1)
    normalized_cm = np.zeros_like(cm, dtype=float)
    for i in range(len(class_names)):
        if row_sums[i] > 0:
            normalized_cm[i, :] = cm[i, :] / row_sums[i]

    plt.imshow(normalized_cm, interpolation='nearest')
    plt.title(rename)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, labels = [0, 1, 2, 3], rotation=45)
    plt.yticks(tick_marks, labels = [0, 1, 2, 3])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(rename + '-confusion-matrix.png')
    plt.show()


def calculate_auc(true_labels, all_preds, num_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        true_binary = np.array([1 if label == i else 0 for label in true_labels])
        pred_scores = np.array([pred[i] for pred in all_preds])
        fpr[i], tpr[i], _ = roc_curve(true_binary, pred_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
    for i in range(len(class_names)):
        if i in fpr and i in tpr:
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=lw,
                     label='ROC curve of class{0} (AUC area = {1:0.2f})'.format(i, roc_auc.get(i, 0.0)))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Sensitivity')
    plt.ylabel('Specificity')
    plt.title(rename)
    plt.legend(loc="lower right")
    plt.savefig(rename + '-ROC.png')
    plt.show()


def main():
    true_labels, pred_labels, all_preds = evaluate_model(model, validation_dir, class_name_to_index, conf_threshold=0.05)

    accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
    print(f'Accuracy: {accuracy:.2f}')

    plot_confusion_matrix(true_labels, pred_labels)

    fpr, tpr, roc_auc = calculate_auc(true_labels, all_preds, len(class_names))
    plot_roc(fpr, tpr, roc_auc)


if __name__ == '__main__':
    main()
