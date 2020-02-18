# Fatigue-Detection

This repo consists of:

<b> PART 1: <br> </b>
A MATLAB application code (Video to EAR.m) which inputs a video and extracts eye-blinks of the user, using a statistical 'EAR (Eye Aspect Ratio)' custom metric based method.
It is an application extension of the works of [1] and [2].

<i> References: <br> </i>
[1] Incremental Face Alignment in the Wild. A. Asthana, S. Zafeiriou, S. Cheng and M. Pantic. In CVPR 2014.
[2] Real-Time Eye Blink Detection using Facial Landmarks. Tereza Soukupova and Jan Cech. In 21st Computer Vision Winter Workshop 2016.<br>


<b> PART 2: <br> </b>
A custom Convolutional Neural Network (VGG based) to train a drowsiness/fatigue detection model based upon input face images.
(Python Script: vgg_face_drowsiness.py, using Keras with TensorFlow as backend).
