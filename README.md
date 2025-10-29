# Face-Attendence-System-using-OpenCV-
To run this project, you will need to install the **opencv-python**, **albumentations**, and **numpy** libraries. You can install all of them at once by running this command in your terminal: `pip install opencv-python albumentations numpy`.
How to Use
Follow these steps in order to run the system.

Step 1: Register Users and Generate Embeddings
This first script (Final_training.py) is used to add new users and to create the embedding database.

Run the Final_training.py script from your terminal:

Bash

python Final_training.py
Part A: Registration Menu You will be given a choice:

Choose 1 (Webcam): The script will ask for your name and ID. It will then activate your webcam, take a picture, and save it (along with 20 augmentations) into a new folder: Data/<Your_Name>/. 

Choose 2 (Folder): The script will look for a folder named Register. It will process any images inside (e.g., Lalith_123.jpg), create augmentations, and save them to Data/Lalith/.

Part B: Embedding Generation

After you finish the registration step (e.g., you take your photo or choose "No"), the script will automatically continue.

It will scan every subfolder in the Data directory (e.g., Data/Lalith, Data/Ayush).

It will generate embeddings for all 21 images for each person and save them into a new embeddings folder (e.g., embeddings/Lalith/Lalith_embeddings.npy).

You must re-run this Final_training.py script every time you add a new user to update the embeddings database.

Step 2: Recognize a User (Live Test)
After you have registered at least one user and generated the embeddings, you can run the testing script.

Run the Final_Testing.py script from your terminal:

Bash

python Final_Testing.py
The script will:

Load all the .npy embedding files from the embeddings folder.

Activate your webcam for a single capture. 

Compare the face it sees to all the embeddings it loaded.

Print a final result, such as --- RESULT: Face Matched: Lalith --- or --- RESULT: Face Not Matched ---. 

If the match fails, the test photo will be saved in the False_Cases folder.

Project Directory Structure
This is what your project folder should look like (the Data, embeddings, and False_Cases folders are created automatically by the scripts):

Plaintext

/Your_Project_Folder/
│
├── Data/                   <-- AUTO-CREATED: Stores all user images
│   ├── Person_A/
│   │   ├── Person_A_123_photo.jpg
│   │   └── Person_A_123_augmented_1.jpg
│   └── Person_B/
│
├── embeddings/             <-- AUTO-CREATED: Stores generated .npy embeddings
│   ├── Person_A/
│   │   └── Person_A_embeddings.npy
│   └── Person_B/
│
├── False_Cases/            <-- AUTO-CREATED: Stores failed recognition images
│   └── false_case_NoFaceDetected_....jpg
│
├── Register/               <-- (Optional) Add batch images here for registration
│   └── Person_C_456.jpg
│
├── Final_training.py       <-- SCRIPT 1: Run this to add users & create embeddings
├── Final_Testing.py        <-- SCRIPT 2: Run this to test recognition
│
├── face_detection_yunet_2023mar.onnx   <-- REQUIRED MODEL
└── face_recognition_sface_2021dec.onnx  <-- REQUIRED MODEL
