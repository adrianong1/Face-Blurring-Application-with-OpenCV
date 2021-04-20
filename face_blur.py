import sys
import cv2

# Functionality of program depends on user argument
imageOrLivePath = sys.argv[1]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Find Faces
def find_faces(grey):
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return faces

# Blur Faces
def blur_faces(faces, frameOrImage):
    for (x, y, w, h) in faces:

        factor = 2.0
        kernelW = int(w/factor)
        kernelH = int(h/factor)
        if kernelW%2 == 0:
            kernelW -= 1
        if kernelH%2 == 0:
            kernelH -=1

        frameOrImage[y:y+h, x:x+w] = cv2.GaussianBlur(frameOrImage[y:y+h, x:x+w], (kernelW, kernelH), 0)

# Split into either a photo of webcam
if imageOrLivePath == "live":       # Webcam
    vidCap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        returnVal, a_frame = vidCap.read()
        grey = cv2.cvtColor(a_frame, cv2.COLOR_BGR2GRAY)
        faces = find_faces(grey)
        blur_faces(faces, a_frame)

        cv2.imshow('Live', a_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidCap.release()
    cv2.destroyAllWindows()

else:       # Image
    # Read the image
    image = cv2.imread(imageOrLivePath)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = find_faces(grey)
    blur_faces(faces, image)

    print("Found {0} face(s)!".format(len(faces)))

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
