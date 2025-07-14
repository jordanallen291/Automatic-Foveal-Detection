# Automatic-Foveal-Detection
 jordanallen291/desktop-tutorial

# Introduction
Over the past few years, we have developed techniques to maintain functional responses in the post-mortem human eye after death (Abbas, Fatima et al. “Revival of light signalling in the postmortem mouse and human retina.” Nature vol. 606,7913 (2022): 351-357. doi:10.1038/s41586-022-04709-x). However, to maintain longer periods of retinal function, biopsy sampling had to be done in total darkness and as quickly as possible. This, compounded by the need for precision sampling within 3 mm, presented a major challenge.
In response to this, we developed a program to interface with the Heidelberg Spectralis OCT hardware and HEYEX software to automatically locate the desired sampling area and send signals to attached stepper motors to quickly align the biopsy punch. This project invovled several custom made parts using Fusion 360 and an object detection model built on Ultralytics YOLO v8 architecture.

# Organization
Within this repository, you'll find the mechanical drawings and components used to adapt the traditionally upright clinical Heidelberg Spectralis to a more accesible transverse orientation, suitable for analyzing _ex vivo_ specimens.

All relevent Python code used to train and test the object detection model and to sample the specimen are included, along with the training data.

# Future Directions
This project is ongoing and will continue to be updated periodically with refined code. The next step is to integrate the motors for automatic alignment.
