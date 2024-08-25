# TinyML Design Contest 2022: Award Winning Solution
This repository contains the winning design from the 2022 TinyML Design Contest, which tasked participants with developing a binary classification model to discriminate life-threatening ventricular arrhythmias (VAs) from non-VAs using single-lead IEGM recordings. The model needed to excel in detection precision, memory occupation, and inference latency, making it suitable for deployment on resource-constrained microcontrollers.

## Project Overview
The core challenge was to classify ventricular arrhythmias like Ventricular Tachycardia (VT), Ventricular Fibrillation (VF), and Ventricular Flutter (VFt) from IEGM signals. The dataset comprised 24,000 training samples and 5,000 test samples, each representing a 5-second IEGM recording.

## Key Features
Signal Processing and Feature Extraction: Utilized techniques like QRS complex detection using morphological filtering, wavelet transforms, and principal component analysis (PCA) to enhance feature extraction from the ECG signals.
Model Development: Explored various models including Convolutional Neural Networks (CNNs), Autoencoders, and traditional classifiers like Logistic Regression and Support Vector Machines (SVM). The CNN-based models, enhanced by Neural Architecture Search (NAS), provided promising results.
Hardware Deployment: The models were deployed on an STM32 NUCLEO-L432KC board, which features an ARM Cortex-M4 core. The deployment pipeline included optimization techniques to reduce memory footprint and improve latency.

## Results
Accuracy and Precision: The optimized Autoencoder model achieved a test accuracy of 94.898%, while the IEGMNet model from the organizers reached 96.391% training accuracy. Further optimizations led to models surpassing 96% accuracy with minimal resource usage.
**Latency**: The best-performing models demonstrated an inference latency of approximately 0.252 seconds on the target hardware, a significant improvement over earlier versions.
**Model Compression**: The use of model quantization reduced the model size by over 40%, enabling deployment on the microcontroller with minimal loss in accuracy.
**Deployment Success**: The logistic regression model with peak extraction was successfully deployed with an F1-β score of 0.9628, showcasing the balance between precision and recall in real-time execution on the hardware.
Repository Contents
**Model Implementations**: Python scripts and ONNX models used for training and deployment.
**Deployment Scripts**: Code for deploying the models onto the STM32 board, including memory optimization and execution instructions.
**Experimental Results**: Detailed logs and performance metrics from various models tested during the contest, with insights into the trade-offs made between accuracy, latency, and memory usage.
## Requirement:
### For training:
* scikit-learn
* pytorch

### For deployment on board:
* MDK5


# How to train our model
We implemented custom feature extraction and used logistic regression from scikit-learn.
the training script is present in `model_training_design\train_model.py`
Our implementation is based on the feature extraction function implemented in the train_model.py file.
The best intercept from the logicistic regression are extracted and used for deployment on the board.

To run the training, simply run the python script `train_model.py` with following parameters `path_data` and `path_indices`, 
the result will display the Intercept and Coefficients learned from the logistic regression.


# How to validate UBPercept model 

### Note: He have not used X-CUBE-AI for generating our model, rather we implemented our own from scratch.

In the folder `deploy_design`, we have the design implemented in the `main.c` and used the template provided by `TEST_OwnModel.zip`. 
In the project, the file `main.c` contains all the functions needed for classification result.

### Note:  no need to implement `Model_Init()` method is the function as we do not have any neural network to be loaded, rather all the logistic regression intercept and coefficient are hardcoded in the `main.c`,  `predict_logistic_reg_v2()` function.

We only impelemnted `aiRun` function to inference the input IEMG segment. The rest of the code, including data reception, data transmission and serial communication, is retained as a template. 


Use the same steps as defined in `Load Program to Board` section of [README-Cube.md](https://github.com/tinymlcontest/tinyml_contest2022_demo_example/blob/master/README-Cube.md)
Also mentioned below:
## Load Program to Board

1. Connet the boadr to computer.

    <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627203515997.png" alt="image-20220627203515997" style="zoom: 25%;" />

2. Open project in MDK5 and build.

    ![build](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/build.png)

3. Check if the debugger is connected.

    First, click ***Options for Target***.

    ![targets](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/targets.png)

    Then switch to <u>Debug</u> and click ***Settings***.

    <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/debug.png" alt="debug"  />

    If the debugger is connected, you can see the IDCODE and the Device Name. 

    <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/swdio.png" alt="swdio"  />

    Finally, switch to <u>Flash Download</u> and check <u>Reset and Run</u>

    ![full chip](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/full%20chip.png)

4. Now you can load program to the board.

    ![load](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/load.png)


## Validation

We use usb2micro usb cable to connect the development board and the upper computer to achieve communication. 

<img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220827121203762.png" alt="image-20220827121203762" style="zoom:50%;" />

Afering connect the board to PC, run the `validation.py` , when seeing output like below, press the **reset button** shown in the picture, and the validation will start.

![iShot_2022-08-27_12.04.57](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/iShot_2022-08-27_12.04.57.png)


#
