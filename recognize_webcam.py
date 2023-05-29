import face_recognition
import cv2
from datetime import datetime, date
import pickle
import pandas as pd
import os

def assure_path_exists(path):
    try:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    except:
        print('Error has occured')

def collect_data(std_name, std_course, std_level, status):
    data = {
    'Name':std_name,
    'Course':std_course,
    'Level':std_level,
    'Status':status
    }
    student = pd.DataFrame(data, columns=["Name", "Course", "Level", "Status"])
    return student

def attendance(student):
    path = 'Database/'
    assure_path_exists(path)
    attendance = 'Attendance.csv'
    files = os.listdir(path)
    if attendance in files:
      student.to_csv(path + attendance, mode='a', index=False, header=False)
    else:
      student.to_csv(path + attendance, index=False)
        
    return attendance
# Load the saved encodings
print('Loading features...')
data = pickle.loads(open('face_enc.pickle', 'rb').read())

# Initialize the video capture
video_capture = cv2.VideoCapture(1)

print('Starting webcam!')
name_ = []
date_ = []
time_ = []
status_ = []
while True:
    # Read a single frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame from BGR to RGB for face recognition
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize an empty list to store recognized names
    recognized_names = []

    
    for face_encoding in face_encodings:
        # Compare the face encoding with the known faces
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        # If there is a match, find the index of the matched known face
        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            
            # Loop over the matched indexes and maintain a count for each recognized face
            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Set the name to the one with the highest count
            name = max(counts, key=counts.get)

        recognized_names.append(name)

    # Draw rectangles and labels on the frame
    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        # If the name is "Unknown", label the face as "Intruder"
        if name == "Unknown":
            name = "Intruder"

            now = datetime.now()
            today = date.today()
            now_date = today.strftime("%d-%b-%Y")
            now_time = now.strftime("%H:%M:%S")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            name_.append(str(name))
            date_.append(str(now_date))
            time_.append(str(now_time))
            status_.append('Access denied')
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            now = datetime.now()
            today = date.today()
            now_date = today.strftime("%d-%b-%Y")
            now_time = now.strftime("%H:%M:%S")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            name_.append(str(name))
            date_.append(str(now_date))
            time_.append(str(now_time))
            status_.append('Access granted')

        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
df = collect_data(name_, date_, time_, status_)
att = attendance(df)
# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
