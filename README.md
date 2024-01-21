# DogBreedClassifer
CNN Models to identify dog breeds using the Stanford dogs dataset. (upto 120 breeds)

# Introduction:

The task of predicting the breed of a dog from an input image is a challenging feat, especially when confronted with a diverse set of up to 120 dog breeds. To address this complexity, a range of convolutional neural network (CNN) architectures were trained and evaluated:

- MobileNetV2
- VGG16
- ResNet50
- GoogleNet-InceptionV3
- DenseNet201

For every model, multiple hyperparameters and model architectures were explored, and only some selected ones are discussed here.

The dataset used is a well-known dataset in the field of computer vision, specifically for tasks related to image classification of dog breeds, originally including around 20,000 images across 120 breeds of dogs.

Each model contributes its unique specifications to the undertaking. **MobileNetV2** achieves a test accuracy of 73.17% with a corresponding test loss of 1.53. In contrast, **VGG16** exhibits a test accuracy of 61.28% coupled with a higher test loss of 2.20. The two iterations of **ResNet50**, denoted as Test and Final, demonstrate test accuracies of 69.21% and 76.04%, respectively, with corresponding test losses of 1.07 and 1.08. **GoogleNet-InceptionV3** excels with a test accuracy of 79.91% and a test loss of 0.82. **DenseNet201_Test** achieves a test accuracy of 82.82% with a test loss of 0.55, while **DenseNet201-Final** emerges as the top performer with an impressive test accuracy of 85.77% and a minimal test loss of 0.45.

### Final Model Chosen - DenseNet201 with a test accuracy of 85.77% and a test loss of 0.45

The central objective of this endeavor is to employ these diverse models, each with its unique hyperparameters and architecture variations, to accurately predict the intricate details of a dog's breed. The evaluation metrics of test accuracy and test loss play pivotal roles in gauging the capabilities of each model in this complex classification task. As we embark on the exploration of these CNN architectures using the Stanford Dogs dataset, the aim is not only to unveil the individual strengths of each model but also to assess their adaptability and efficacy in the real-world scenario of predicting dog breeds from images. This dataset serves as a challenging benchmark, pushing the boundaries of image classification and providing a platform for advancing the state-of-the-art in canine breed recognition.

![image](https://github.com/TheekshithaVaratharajsarma/DogBreedClassifer/assets/129731048/275e9363-3286-4b29-a469-3cafd0fc95c0) ![image](https://github.com/TheekshithaVaratharajsarma/DogBreedClassifer/assets/129731048/807b1466-b846-4af0-abc6-7a8ac073d477) ![image](https://github.com/TheekshithaVaratharajsarma/DogBreedClassifer/assets/129731048/a99e0dea-38a2-4dba-bc1d-e4b78c54ab4d)  ![image](https://github.com/TheekshithaVaratharajsarma/DogBreedClassifer/assets/129731048/fe3a3912-4572-413c-81e5-70930335c3b2)  ![image](https://github.com/TheekshithaVaratharajsarma/DogBreedClassifer/assets/129731048/335cebd4-1464-4b57-a084-2da0f1db17bf)

![image](https://github.com/TheekshithaVaratharajsarma/DogBreedClassifer/assets/129731048/c1fb27f9-3b3b-4323-9420-c0b843797567)







