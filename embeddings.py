from imutils import paths
import face_recognition
import pickle
import cv2
import os
 
#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
imagePaths = list(paths.list_images('Images'))
knownEncodings = []
knownNames = []
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    print('person: ', i, 'out of', len(imagePath))
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # boxes = rgb
    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use
f = open("face_enc.pickle", "wb")
f.write(pickle.dumps(data))
f.close()



# import face_recognition
# import cv2
# import os

# # Load the known faces and encodings
# known_faces_dir = 'test/Muhammad Aliyu/'
# known_faces = []
# known_names = []

# # Load known faces from the directory
# for filename in os.listdir(known_faces_dir):
#     image_path = os.path.join(known_faces_dir, filename)
#     image = face_recognition.load_image_file(image_path)
#     encoding = face_recognition.face_encodings(image)[0]
#     known_faces.append(encoding)
#     known_names.append(os.path.splitext(filename)[0])

# # Initialize the video capture
# video_capture = cv2.VideoCapture(0)

# while True:
#     # Read a single frame from the video capture
#     ret, frame = video_capture.read()

#     # Convert the frame from BGR to RGB for face recognition
#     rgb_frame = frame[:, :, ::-1]

#     # Find all the faces in the frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     # Initialize an empty list to store recognized names
#     recognized_names = []

#     for face_encoding in face_encodings:
#         # Compare the face encoding with the known faces
#         matches = face_recognition.compare_faces(known_faces, face_encoding)
#         name = "Unknown"

#         # If there is a match, find the index of the matched known face
#         if True in matches:
#             matched_idx = matches.index(True)
#             name = known_names[matched_idx]

#         recognized_names.append(name)

#     # Draw rectangles and labels on the frame
#     for (top, right, bottom, left), name in zip(face_locations, recognized_names):
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Face Recognition', frame)

#     # Exit the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all windows
# video_capture.release()
# cv2.destroyAllWindows()
