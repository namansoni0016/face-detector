# Importing OpenCV
import cv2
from random import randrange

# Loading some pre-trained data of face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Haar Cascade algorithm basically means, classifier for object detection

# To capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Grayscale for webcame
    grayscaled_web = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Face (that includes different sizes)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_web)
    #print(face_coordinates)

    # Draw rectangles around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

    # For showing video
    cv2.imshow('My Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed (using ascii value for Q/q)
    if key==81 or key==113:
        break
# Release the VideoCapture object
webcam.release()


"""
#FOR IMAGE 


# Choose images to detect face
img = cv2.imread('virat.png')
img2 = cv2.imread('rdj.png')
img3 = cv2.imread('group.png')

# Haar cascade algorithm only takes gray scale images, so now we need to change images into black & white
grayscaled_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect Face (that includes different sizes)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#print(face_coordinates)

# Draw rectangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img2, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

# Display the image on screen
cv2.imshow('My Face Detector', img2)

# To display the image you have to hold the execution of your program, waitkey() basically stops the execution of your program until you press any key
cv2.waitKey()
"""