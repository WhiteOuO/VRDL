# NYCU Computer Vision 2025 Spring HW1
StudentID:110550128  
Name:蔡耀霆
## Introduction:
I have chosen to use pretrained weights for image recognition, which will give me a good starting point, and then let the model gradually adapt to my dataset and learn. I plan to implement some data augmentation so that each image in my training set can be transformed through different image processing techniques to provide more comprehensive features for that class, such as horizontal flipping, slight changes in brightness and hue, and cropping the image.
In addition to image processing techniques, I also plan to use methods to exclude outliers from the training data. I believe this will help the model better learn the common features of the class from the normal data.
Furthermore, I will also apply a learning rate adjustment strategy, some model layer addition and some techniques to help my model progress more effectively.
## How to install
pip install -r requirements.txt  
## Performance snapshot
![image](https://github.com/user-attachments/assets/65d7e8db-25e2-4677-a3cc-5d9a2c880974)  
![image](https://github.com/user-attachments/assets/1ef716db-2a2a-43d1-83d0-0853376d5464)
