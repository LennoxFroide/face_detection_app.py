#This code uses the pre-trained haarcascade algorithm from openCV.
#We have to install the openCv module as well.
import cv2 as cv
from random import randrange



#Pre_trained data on face frontals.
trained_face_data = cv.CascadeClassifier("haarcascade_frontalface_default.xml") 

"""Choose an image to detect faces in. We use the image read function from openCv. The image is imported as an array since an image is an array of pixels."""

img =cv.imread('jesper.jpeg')


#We make the image to geryscale which is easier for the algorithm to handle. 

#The cvtColor function takes an image and the format to change it to
greyscale_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

"""To detect the faces.The detect multiscale function means that no matter the scale of the image i.e big or small, it will still be detected.This function returns the upper left (2) and bottom right (2)coordinates of the rectangles surrounding the faces which will allow the algorithm to draw rectangles on the image or frame of the video with the face that we're trying to detect."""

face_coordinates = trained_face_data.detectMultiscale(greyscale_img)

#Draw a rectangle around the face using rectangle function. 

"""We use the original colour image here.We will use a  Rectangle function with 5 arguments. It takes the coloured image, a tuple of the (upper left hand) coordinate and another for the (upper left+lower right hand)coordinate. It also allows you to select a colour from BGR (0,255,0) would mean that we remove the blue and red and set green to 255 so our rectangle will have a green border. The last argument it takes is the thickness of the rectangle set as an integer."""

#We are assigning the coordinate of the face to variables x,y,w,h so that we could use it for multiple faces.

#The for loop allows us to loop through the multiple faces in the array/list

for (x,y,w,h) in face_coordinates:


  cv.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),5)



#This will display the image. It take the name of the window and an image to be displayed on the window as arguments.

###The comment below could be converted back to code to view the image.
#cv.imshow('Jesper Fayeh, Netflix Shadow and Bone',img)



#Let's print face face_coordinates

#print(face_coordinates)


#This ensures that the compiler waits so that we can view the image.
cv.waitKey()





"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

"""This next section is a code to detect face in a video!!!
"""


#We use the VideoCapture function to capture a video from our webcam or a video file.

webcam = cv.VideoCapture(0)

#We use a while loop to make it loop over all the frames forever until we stop the webcam.

while True:
  #To read the current frame, we use read function. It retrune a boolean to indicate whether the read was successful or nt and the frame

  successful_frame_read, frame = webcam.read()

  #Converts webcam frames to greyscale
  greyscale_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

  #Face detection 
  face_coordinates = trained_face_data.detectMultiscale(greyscale_img)

  #Draw our rectangle
  for (x,y,w,h) in face_coordinates:

    cv.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),5)


  #To display our image.
  cv.imshow('Webcam_image_face_detector',frame)


  #We set waitkey to 1 to make sure that it will switch automatically to the next frame after every millisecond
  key = cv.waitkey(1)

  """To ensure that we can quit out of the webcam whenever we want to, we have to create a stop key. We use s key for stop. To do this we use the ASCII 115 for s and 83 for S
  """

  if key==83 or key==115:
    break

#We clean up our Code/ we deallocate the memory.
webcam.release()




print('Face detector app complete!!')


