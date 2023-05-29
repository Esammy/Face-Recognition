import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#find path of xml file containing haarcascade file
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
print('Encoding loaded!')
#Find path to the image you want to detect face and pass it here
img_path = 'Obinna.jpg' #No
# img_path = 'test/Muhammad Aliyu/IMG_4175.jpg' #Yes
# img_path = 'test/Muhammad Ndagi/IMG_4243.jpg' #Yes
# img_path = 'test/Abdulazeez Taiwo/IMG_4045.jpg' #Yes
# img_path = 'test/Nura Muhammad Idris/IMG_4107.jpg' #Yes

image = cv2.imread(img_path)
print(image)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#convert image to Greyscale for haarcascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.3,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

if faces != 0:
    print('face detected!')
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image_frame = image[y:y + h, x:x + w]

        # the facial embeddings for face in input

        encodings = face_recognition.face_encodings(image_frame)
        names = []
        print('Encoding:', encodings)
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
            #Compare encodings with encodings in data["encodings"]
            #Matches contain array with boolean values and True for the embeddings it matches closely
            #and False for rest
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            #set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                    #set name which has highest count
                    name = max(counts, key=counts.get)
        
        
                # update the list of names
                names.append(name)
                # loop over the recognized faces
                for ((x, y, w, h), name) in zip(faces, names):
                    # rescale the face coordinates
                    # draw the predicted face name on the image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    5.0, (0, 255, 0), 5)
                new_size = (500, 500)
                resize_img = cv2.resize(image, new_size)
                cv2.imshow("Frame", resize_img)
                cv2.waitKey(0)  
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, 'Unknown', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    5.0, (0, 255, 0), 5)
                new_size = (500, 500)
                resize_img = cv2.resize(image, new_size)
                
                # cv2.imshow("Frame", resize_img)
                # cv2.waitKey(0)

else:
    print('No face')