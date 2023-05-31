import cv2
import os

def assure_path_exists(path):
    try:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    except:
        print('Error has occured')
        

img_dataset = ['Dataset/']
for folder in img_dataset:
    assure_path_exists(folder)

def detect_face():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    std_f_length = len(os.listdir('Dataset/'))
    
    images_folder = os.listdir('Dataset')
    for folder in images_folder:
        images = os.listdir('Dataset'+'/'+str(folder))


        print('\nDetecting faces in ' +str(folder) + ' folder')
        
        for image in images:
            img = ('Dataset'+'/'+str(folder)+'/'+str(image))
            assure_path_exists('Images/'+str(folder)+'/')
            img_num = len(os.listdir('Images/'+str(folder)+'/'))
            img_name = img_num + 1
            #imgPath = 'images/Johnmark/1.jpg'

            imag = cv2.imread(img)
            # print(imag.shape)
            
            

            
            gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            print ("Found {0} faces!".format(len(faces)))

            
            for (x, y, w, h) in faces:
                cv2.rectangle(imag, (x, y), (x+w, y+h), (0, 255, 0), 2)

                image_frame = imag[y:y + h, x:x + w]

                new_size = (128, 128)
                resize_img = cv2.resize(image_frame, new_size)
                

                cv2.imwrite('Images/'+str(folder)+'/' + str(img_name)+ '.jpg', resize_img)
            
            # cv2.imshow("Faces found", resize_img)
            # cv2.waitKey(0)


detect_face()