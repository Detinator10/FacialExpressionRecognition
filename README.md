# Facial Expression Recognition

This program reads input from the computer's webcam, examines a user(s) face, and uses a Convolutional Neural Network to recognize facial expressions. It is written in Python and uses the OpenCV and Tensorflow libraries. The program uses OpenCV to identify and crop faces out of webcam input. Then, using the CNN created in Tensorflow, this cropped face image is fed into the network to produce a prediction of one of seven emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, or Surprised). The prediction is then displayed above the user(s) face in a window which shows the webcam feed.

## Training
The CNN written in Tensorflow can be trained using the train.py file. This file trains the algorithm using the Facial Expression Recognition 2013 dataset from Kaggle (https://www.kaggle.com/deadskull7/fer2013) and validates its training with the test portion of the dataset. It plots learning curves of the loss and accuracy over each epoch using pyplot and then saves the model to model.h5 (this allows the model to be reloaded later without requiring training to be run again). 

## Testing
Testing is done during/after the training process, but can be run separately using the test.py file. This will run the loaded model over the test dataset and print data about its accuracy and loss.

## Instructions/How to run

If you would like to train the algorithm yourself, you may run the train.py file to first train the algorithm (this may take quite some time but can be sped up with GPU assistance). To run the program itself, simply run the gui2.py file. This will load the model from the saved file, open a window showing video from your computer's webcam, and start making predictions about your facial expression both above your head in the window and in the python console. To quit the program, simply press 'q' on your keyboard.

## Performance Metrics

This project is currently in development, but once I have finished I will update this with the final performance metrics for the program.
