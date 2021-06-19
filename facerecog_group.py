import cv2
import os

imagePath = os.path.abspath("Images/group1.jpg")
cascPath = os.path.abspath("haarcascade_frontalface_default.xml")

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Filename 
filename = 'myconvertedimage_group.jpg'
  
# Saving the image 
cv2.imwrite(filename, image) 