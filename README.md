# face_detection_app.py
This is a face detection app using python. The first section deteccts faces in a photo and the second section detects faces in a video. It could a video stream from a webcam or a video file.

To use this code you need to install the OpenCv module.

I used the haarcascade algorithm provided by OpenCv on github. This contained a pre-trained neural network that was trained using supervised machine learning to identify faces and non faces.

It is called a cascade as it’s a chain of machine learning code through which an image is passed.

Each square of each image is passed through the machine learning code and it cascades through each section of code. If it passes all cascades and gets to the bottom, then algorithm knows that it looks close enough to being a face thus it is identified as one.

The algorithm uses 5 rudimentary Haar features which include Edge features, Line features and Four-rectangle features which are basically black and white rectangles that are layed over the image and then the algorithm compares whether there’s any relationship between the black and whie portions.

Multiple features are used together and layered over one another until a face is formed, if there’s a relationship.


