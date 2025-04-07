#  Elevating Computer Vision: The Combined Strengths of ArcFace, VGGFace2, and OpenFace 2.0

## Introduction
This project integrates many AI Computer Vision Algorithms including: OpenFace, InsightFace, ArcFace, and DeepFace. Through this integration the program creates a facial detection and recognition algorithm where the user can input any image and it will detect if there is a human face present as well as the name ,if the programs previous embedded images match, age, and sex. The environment best used is Google Colab using jupiter notebook if you want to use a another python environment, then you may need to adjust some of the code to fit that environment.

## How to Run the Experiments
### Prerequisites
- Google Colab Environment
- Utilize these Python Libraries: dlib, deepface, insightface, OpenCV, numpy

### Steps
1. **Open the Jupiter Notebook on Google Colab**. If you want to use a different Python environment you will have to change some structure to the code.
2. **Mount your Google Drive** if you want to upload your own images to embed, so they can be stored locally to the Google Colab Environment.
   ```
   from google.colab import drive
   drive.mount('/content/drive/')
   ```
3. **Install the proper dependencies from the example code block**:
   ```
   !pip install dlib
   !pip install deepface # Uses VGGFace2 for age and sex classifications
   !apt-get install -y cmake
   !pip install insightface # used for ArcFace
   !pip install mxnet
   !pip install onnxruntime
   !pip install opencv-python
   ```
4. **Import the Necessay Libraries**: 
   ```
   import cv2
   import glob
   import numpy as np
   import os
   from google.colab import files
   from io import BytesIO
   from PIL import Image
   from google.colab.patches import cv2_imshow
   from deepface import DeepFace
   ```
5. **Setup the OpenFace Environment**: I used git clone for a given OpenFace repository
   ```
   !git clone https://github.com/cmusatyalab/openface.git
   %cd openface
   !python setup.py install
   import openface
   import dlib
   dlib_model_dir = '/content/openface/models/dlib'
   
   if not os.path.exists(dlib_model_dir):
       os.makedirs(dlib_model_dir)
   
   model_file_bz2 = os.path.join(dlib_model_dir, 'shape_predictor_68_face_landmarks.dat.bz2')
   model_file_dat = os.path.join(dlib_model_dir, 'shape_predictor_68_face_landmarks.dat')
   
   if not os.path.exists(model_file_dat):
       if not os.path.exists(model_file_bz2):
           !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O {model_file_bz2}
       !bzip2 -d {model_file_bz2}
   else:
       print("dlib model file already exists.")
   
   dlib_face_predictor = os.path.join(dlib_model_dir, 'shape_predictor_68_face_landmarks.dat')
   openface_model_dir = '/content/openface/models/openface'
   network_model = os.path.join(openface_model_dir, 'nn4.small2.v1.t7')
   
   align = openface.AlignDlib(dlib_face_predictor)
   ```
6. **Setup ArcFace**:
   ```
   from insightface.app import FaceAnalysis
   from scipy.spatial.distance import cosine

   app = FaceAnalysis()
   app.prepare(ctx_id=-1, det_size=(640, 640))
   ```
7. **Run Each Code Cell Sequentially**: These are the functions utilized for embedding and image processing
8. **Embed Images**: This function is optional if you do not wish to use facial recognition. This function call will ask you to either manually input images of a person or people, or you will have to hardcode the image files you want to embed.
   ```
   upload_for_embedding()
   ```
9. **Testing:** Using the function call below will allow you to test the facial recognition and/or detection of any uploaded image(s). This will output any facial detection and the image's sex and age if a face is detected. The user will also have the option to take a photo if they want to capture a photo using their device camera.
    ```
    Test_Process()
    ```
   
## Code Overview
### Code Structure
The code provided in the Notebook goes through **five** essential steps:
1. **User Uploads images of certain people or person**: The first thing the code will ask of the user is to upload images of a designated person or people to embed. It will first give you the option to have the image files uploaded through hardcoding by adding the path of your directories. The other option is to manually upload the images if you want. For example, I utilized the hardcode option to copy six directory paths of each of the friends cast, where each directory has 10 images of each person. Through these images, the code will use ArcFace embedding so it can recognize any of the designated members when uploading the test image. The code block below shows where I manually added the paths of face images for the cast of friends. If you want to use your own dataset you will need to change this section:
   ```
    elif option.lower() == "hardcode":
     # Define the list of people and their corresponding folder names
     people_folders = [
         ("Jennifer Aniston", "Jennifer_Aniston"),
         ("David Schwimmer", "David_Schwimmer"),
         ("Courtney Cox", "Courtney_Cox"),
         ("Matthew Perry", "Matthew_Perry"),
         ("Lisa Kudrow", "Lisa_Kudrow"),
         ("Matt LeBlanc", "Matt_LeBlanc")
     ]
     for person_name, folder_name in people_folders:
      folder_path = os.path.join("/content/drive/MyDrive/Colab Notebooks/AI_Project/Friends_Cast", folder_name, "Embedding")
   ```
2. **User Uploads Test Image**: The last code cell "test_process" will ask the user to either input image(s) or take a photo to go through the facial recognition and detection process.
3. **OpenFace Process**: The image is then processed through the OpenFace libraries and dependencies for basic facial detection.. When the image is output, you can see if OpenFace detected a face by a blue square around the test image's face.
4. **ArcFace Recognition**: After processing through OpenFace it will then go through insightface enhanced facial recognition to detect and outline facial features such as jawline or eyes. Then through insightface, ArcFace is used to compare the test images to the previous embedded images to see if the face is recognizable. It will either output on the image "unknown" or the recognized name.
5. **DeepFace Demographic Classifications**: The test image is then processed through the DeepFace library utilizing the VGGFace model. Here it will estimate the detected face age and sex. Then the image will output with all of its classifications. The image below shows an example:
   ![Result](Screenshot 2023-11-20 205808)

### Code from Other Repositories
- **OpenFace Setup**: The setup and initialization code for OpenFace is derived from the [OpenFace GitHub Repository](https://github.com/cmusatyalab/openface).
- **InsightFace Integration**: The integration of InsightFace for ArcFace is based on code from the [InsightFace GitHub Repository](https://github.com/deepinsight/insightface).

### Modified Code
- **CV2**: For image reading I used CV2, where I found code from numerous external sources online including this link: https://reny-jose.medium.com/an-introduction-to-opencv-using-google-colab-notebooks-part-1-8600c597034b
- **DeepFace**: Utilized Google Colabs installed dependencies for DeepFace. Code was structured from many external sources such as StackOverflow
  
### Original Code
- **Webcam Capture**: The `take_photo` function for capturing images from the webcam is original and specifically designed for use in Google Colab.
- **User InterFace**: The code to allow the user to upload images or any other fun structural use for the user.

## Dataset Description
As mentioned earlier, when embedding images, I used 60 images that represented the cast of friends. This dataset will need to be changed for the user's own images they wish to embed. Alternatively, you can just choose the manual option to upload files to the google colab environment for them to be embedded.

## Additional Notes
- **Face Recognition Threshold**: The threshold for face recognition is hardcoded at 0.58. The value was determined through a lot of trial and error. The number may vary depending on how many images are used to embed per person.
- **Demographic Inaccuracies**: The DeepFace library that uses VGGFace as the model has some inaccuracies when it comes to detecting the age and sex of the processed image. If you want more accurate results I would recommend using a better model such as UTKFace.
- **Embedding Dataset**: If you want to embed your own images, make sure to change the file paths in the correct section of your code. These file paths correspond to my Google Drive.

## Video Link
Link for Explanation: https://youtu.be/Q3yHgwCev9E
