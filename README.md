# **Hyperpigmented Skin Disease Classification Using Deep Learning Algorithms**

## Introduction
There are growing numbers of significant skin disorders, including skin pigmentation. It states that skin color is determined by the amount of melanin produced by the body. The two main categories of skin pigmentation including:
- Hyperpigmentation, in which pigment seems to overflow.
- Hypopigmentation, in which pigment appears to decrease.

However, many skin conditions share characteristics, making it difficult for dermatologists to correctly diagnose their patients. Consequently, the accurate early detection of skin disorders and the diagnosis of dermatoscopy pictures can be greatly aided by machine learning and deep learning approaches.

## Objective
To find the most effective deep learning technique for picture identification was investigated in order to diagnose hyperpigmented skin diseases.

## Deep Learning Models
The following pretrained deep learning models were evaluated:
- YOLO (You Only Look Once)
- DenseNet201
- GoogLeNet
- InceptionResNetV2
- MobileNet

## Results
The summary of each model training and testing accuracy rates and AUC Score of each model:

| Model                | Training Accuracy | Test Accuracy  |AUC Score|
|----------------------|-------------------|----------------|----------------|
| YOLO                 | 97.43%            | 97.56%         | 0.97           |
| DenseNet201           | 100%              | 87.18%         |0.99           |
| GoogLeNet            | 93.8%             | 87.18%         |0.90          |
| InceptionResNetV2    | 98.77%            | 89.74%         |0.94           |
| MobileNet            | 100%              | 79.49%         |0.98           |

### Best Performing Model
- **DenseNet201** achieved the highest accuracy and AUC values, making it the top-performing model. However, **YOLO** was also highlighted for its performance based on confusion matrix analysis.

## Conclusion
Briefly state the findings of the project. DenseNet201 was the most effective model, but Yolo has the most stable outcome performance, however more research is needed to enhance its clinical applications.

## Future Work
Outline the steps for improving the model:
- **Dataset Expansion**: Mention the importance of using a larger dataset for better model generalization.
- **Hybrid Models**: Discuss the potential of exploring hybrid models to further improve accuracy.
- **Clinical Testing**: Highlight the importance of testing these models in real-world clinical settings.
