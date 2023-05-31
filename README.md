# Face Recognition Program
Welcome to the Face Recognition Program, a sophisticated computer vision solution designed to identify faces in images. This program utilizes cutting-edge techniques and algorithms to deliver accurate and efficient face recognition capabilities. Let's dive into the details of this powerful system.

## Files
The program consists of four major files, each serving a specific purpose. Here's an overview of each file:

`detect_face.py`: This file contains the implementation of the face detection algorithm. It leverages the power of OpenCV and uses the `haarcascade_frontalface_default.xml` file to detect faces in images. The detected faces are then resized to a standardized shape of 128 x 128 for further processing.

`recognize_image.py`: This file focuses on face recognition functionality. It employs a sophisticated machine learning model trained on face images to accurately recognize and classify the detected faces. By assigning labels to each recognized face, the program enables you to identify individuals based on their unique facial features.

`recognize_webcam.py`: Similar to recognize_image.py, this file offers face recognition capabilities. However, it specializes in real-time recognition by capturing input from your webcam. By analyzing the captured video feed, the program swiftly classifies individuals based on the labels associated with them.

`embeddings.py`: This file plays a crucial role in the training process. It extracts the essential facial features required for training the face recognition model. By processing the images stored in the Dataset folder, it generates facial embeddings that form the basis of accurate recognition.

## Performing Recognition
To perform face recognition using this program, please follow these steps:

1. Clone the repository to your local machine to gain access to the program's source code.

2. Ensure that you have all the necessary dependencies installed. You can find a comprehensive list of the required packages in the `requirements.txt` file provided.

3. Create a folder named `Dataset` within the project directory. Inside this folder, create subfolders, each named after the individuals you want the algorithm to recognize.

4. Populate each subfolder with a diverse range of 10 to 50 images of the respective individuals. These images will be used to train the face recognition algorithm.

5. Open the project folder, specifically the `Face-Recognition` folder, in your preferred development environment, such as Visual Studio Code (VSCode).

6. Run the `detect_face.py` file. This script will automatically extract faces from each image in every individual's folder. The extracted faces will be resized and saved in the Image folder for subsequent processing.

7. Execute the `embeddings.py` file to extract crucial facial features from the preprocessed images. These features will serve as the basis for training the face recognition model.

8. Finally, run the `recognize_webcam.py` file. This script will access your webcam, enabling real-time face recognition. Make sure to adjust the parameter in the line `video_capture = cv2.VideoCapture(1)` according to your camera configuration. Use zero `(0)` for a built-in webcam or one `(1)` for an external camera.

Remember, this program offers great flexibility for customization. Feel free to explore and experiment with various settings, models, and datasets to enhance the performance and accuracy of the face recognition system.

If you encounter any difficulties or have further questions, please don't hesitate to seek assistance. We hope you find this Face Recognition Program valuable and enjoy using it!
