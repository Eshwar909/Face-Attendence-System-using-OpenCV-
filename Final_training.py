import cv2
import os
import albumentations as alb
import time

# =============================================================================
# FUNCTION 1: INTERACTIVE WEBCAM REGISTRATION
# =============================================================================
def get_user_details():
    """
    Main function to get user's decision, details, and a confirmed photo
    via webcam.
    """
    print("\n--- Starting Interactive Webcam Registration ---")
    while True:
        enquiry = input("Do you want to take a picture or not? (Yes/No): ").lower()

        if enquiry in ["yes", "y", "ye", "es"]:
            while True:
                name = input("Enter your name: ")
                if not name:
                    print("No name given.")
                    continue
                if name.replace(' ', '').isalpha():
                    break
                else:
                    print("Give me a valid name.")

            while True:
                ID = input("Enter your ID (No special characters are allowed): ")
                if not ID:
                    print("No ID given.")
                    continue
                if ID.isalnum():
                    break
                else:
                    print("Invalid input. Please enter only numbers and letters for your ID.")

            # --- Photo Capture and Augmentation Logic ---
            try:
                base_data_folder = "Data"
                user_folder_path = os.path.join(base_data_folder, name)
                os.makedirs(user_folder_path, exist_ok=True)

                print("\nStarting camera...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open webcam.")
                else:
                    # Loop to allow for retaking the photo
                    while True:
                        print("Taking a picture in 3 seconds...")
                        time.sleep(1)
                        print("2...")
                        time.sleep(1)
                        print("1...")
                        time.sleep(1)

                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to capture frame.")
                            break

                        # --- NEW: Save a temporary preview and ask for confirmation ---
                        temp_image_path = "temp_preview.jpg"
                        cv2.imwrite(temp_image_path, frame)
                        print(f"Image captured. A preview has been saved as '{temp_image_path}'.")
                        print("Please open this file to check the image.")

                        confirmation = input("Would you like to keep this image? (Yes/No): ").lower()

                        if confirmation in ["yes", "y"]:
                            # User is happy with the image, proceed.
                            print("Image confirmed. Saving permanently and generating augmentations...")

                            # Define the final path and rename the temp file to save it
                            file_path = os.path.join(user_folder_path, f"{name}_{ID}_photo.jpg")
                            os.rename(temp_image_path, file_path)
                            print(f"Original photo saved in folder: '{user_folder_path}'")

                            # --- Image Augmentation ---
                            print("Now generating 20 augmented images...")
                            augmentor = alb.Compose([
                                alb.HorizontalFlip(p=0.5),
                                alb.RandomBrightnessContrast(p=0.2),
                                alb.RandomGamma(p=0.2),
                                alb.RGBShift(p=0.2),
                            ])
                            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image_resized = image_rgb # Using your latest "no resize" logic

                            for i in range(20):
                                augmented = augmentor(image=image_resized)
                                augmented_image_rgb = augmented['image']
                                augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)
                                aug_file_path = os.path.join(user_folder_path, f"{name}_{ID}_augmented_{i+1}.jpg")
                                cv2.imwrite(aug_file_path, augmented_image_bgr)

                            print(f"Successfully saved 20 augmented images in '{user_folder_path}'.")
                            break # Exit the retake loop

                        else:
                            # User wants to retake the picture
                            print("Discarding image. Let's try again.")
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                            # The loop will now repeat, taking a new picture.

                cap.release()

            except Exception as e:
                print(f"\nAn error occurred while trying to take or augment the photo: {e}")
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

            print(f"\nThank you, {name}. Your details have been recorded.")
            break

        elif enquiry in ["no", "n", "o"]:
            print("Ok, proceeding without taking a picture.")
            break
        else:
            print("Invalid response. Please type 'Yes' or 'No'.")

# =============================================================================
# FUNCTION 2: AUTOMATED BATCH REGISTRATION
# =============================================================================
def register_images_from_folder(input_folder):
    """
    Loops through an input folder, reads each image, and then saves the
    original plus 20 augmented versions to a structured 'Data' folder.
    """
    print("\n--- Starting Automated Batch Registration ---")

    # Define the main data folder
    base_data_folder = "Data"

    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    print(f"Starting batch registration from folder: '{input_folder}'")

    # Loop through every file in the specified input folder
    for filename in os.listdir(input_folder):

        # Check if the file is a common image type
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

            # --- 1. Extract Name, ID, and Set Up Paths ---
            base_filename = os.path.splitext(filename)[0]
            if '_' in base_filename:
                parts = base_filename.rsplit('_', 1)
                if len(parts) == 2:
                    name, ID = parts[0], parts[1]
                else:
                    print(f"Warning: Filename '{filename}' not in 'Name_ID' format. Using base filename as name.")
                    name = base_filename
                    ID = "unknown" # Assign a default ID
            else:
                print(f"Warning: Filename '{filename}' not in 'Name_ID' format. Using base filename as name.")
                name = base_filename
                ID = "unknown" # Assign a default ID

            print(f"  > Processing file: '{filename}' -> Name: '{name}', ID: '{ID}'")
            user_folder_path = os.path.join(base_data_folder, name)
            os.makedirs(user_folder_path, exist_ok=True)
            original_image_path = os.path.join(input_folder, filename)

            # --- 2. Load and Save the Original Image ---
            try:
                frame = cv2.imread(original_image_path)
                if frame is None:
                    print(f"Warning: Could not read image '{filename}'. Skipping.")
                    continue

                file_path = os.path.join(user_folder_path, f"{name}_{ID}_photo.jpg")
                cv2.imwrite(file_path, frame)
                print(f"\nSuccessfully registered '{name}' and saved original photo.")

                # --- 3. Image Augmentation ---
                print(f"Now generating 20 augmented images for '{name}'...")
                augmentor = alb.Compose([
                    alb.HorizontalFlip(p=0.5),
                    alb.RandomBrightnessContrast(p=0.2),
                    alb.RandomGamma(p=0.2),
                    alb.RGBShift(p=0.2),
                ])
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_resized = image_rgb

                for i in range(20):
                    augmented = augmentor(image=image_resized)
                    augmented_image_rgb = augmented['image']
                    augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)
                    aug_file_path = os.path.join(user_folder_path, f"{name}_{ID}_augmented_{i+1}.jpg")
                    cv2.imwrite(aug_file_path, augmented_image_bgr)

                print(f"Successfully saved 20 augmented images in '{user_folder_path}'.")

            except Exception as e:
                print(f"\nAn error occurred while processing '{filename}': {e}")
                continue # Skip to the next image

    print("\n--- Batch Registration Complete ---")

# =============================================================================
# SCRIPT EXECUTION (MAIN MENU)
# =============================================================================
if __name__ == "__main__":

    print("=======================================")
    print("  Face Registration System")
    print("=======================================")
    print("Please choose an option:")
    print("  1: Register a new user (via Webcam)")
    print("  2: Register existing images (from Folder)")
    print("=======================================")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        get_user_details()

    elif choice == '2':
        # Define the name of the folder containing your source images
        INPUT_IMAGE_FOLDER = "Register"
        register_images_from_folder(INPUT_IMAGE_FOLDER)

    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")

    print("\nRegistration script finished.")
# =============================================================================
# CELL 1: Imports and Initial Setup
# =============================================================================
# Purpose: Import all necessary libraries and configure the environment.
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import warnings

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore")
print("Cell 1/4: Libraries imported and environment configured.")
# =============================================================================
# CELL 2: Model File Verification
# =============================================================================
# Purpose: Define the model filenames and verify that they exist in the
#          project folder before proceeding.
# -----------------------------------------------------------------------------
# --- Model Information ---
FACE_DETECTION_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx"

def check_model_exists(model_name):
    """Checks if a model file exists in the current directory."""
    if not os.path.exists(model_name):
        print(f"Error: Model file not found: '{model_name}'.")
        print("Please make sure it is in the same folder as this script.")
        return False
    else:
        print(f"Model file '{model_name}' found.")
        return True

# --- Execute the file check for both models ---
detector_ready = check_model_exists(FACE_DETECTION_MODEL)
recognizer_ready = check_model_exists(FACE_RECOGNITION_MODEL)
print("Cell 2/4: Model file check complete.")

# =============================================================================
# CELL 3: Embedding Generation Function
# =============================================================================
# Purpose: Define the main function that iterates through a folder of images,
#          detects faces, and generates embeddings for each one.
# -----------------------------------------------------------------------------
def generate_face_embeddings(image_folder_path, detector, recognizer):
    """
    Detects faces in all images in a folder, generates embeddings, and saves them.
    """
    if not os.path.isdir(image_folder_path):
        print(f"Error: Source image folder not found at '{image_folder_path}'")
        return

    person_name = os.path.basename(image_folder_path)
    all_embeddings = []

    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(image_files)} images for '{person_name}'. Processing...")

    for filename in image_files:
        image_path = os.path.join(image_folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  - Warning: Could not read '{filename}'. Skipping.")
            continue

        h, w, _ = image.shape
        detector.setInputSize((w, h))
        _, faces = detector.detect(image)

        if faces is not None and len(faces) > 0:
            try:
                face_coords = faces[0]
                aligned_face = recognizer.alignCrop(image, face_coords)
                embedding = recognizer.feature(aligned_face)
                all_embeddings.append(embedding)
                print(f"  - Generated embedding for {filename}")
            except Exception as e:
                print(f"  - Warning: Could not process face in '{filename}'. Error: {e}")
        else:
            print(f"  - Warning: No face detected in {filename}. Skipping.")

    # --- Save the Results ---
    if all_embeddings:
        output_dir = os.path.join("embeddings", person_name)
        os.makedirs(output_dir, exist_ok=True)

        embeddings_path = os.path.join(output_dir, f"{person_name}_embeddings.npy")
        labels_path = os.path.join(output_dir, f"{person_name}_labels.npy")

        labels = [person_name] * len(all_embeddings)

        np.save(embeddings_path, np.array(all_embeddings))
        np.save(labels_path, np.array(labels))

        print(f"\nSuccessfully saved {len(all_embeddings)} new embeddings for '{person_name}'.")
    else:
        print("\nNo new embeddings were generated.")

print("Cell 3/4: Embedding generation function defined.")
# =============================================================================
# CELL 4: Execution Block
# =============================================================================
# Purpose: Load the models and call the main processing function for each
#          person's subfolder found within the main 'Data' directory.
# -----------------------------------------------------------------------------
if detector_ready and recognizer_ready:
    print("\nLoading models...")
    try:
        # Load Face Detector
        face_detector = cv2.FaceDetectorYN.create(
            model=FACE_DETECTION_MODEL, config="", input_size=(320, 320)
        )
        # Load Face Recognizer
        face_recognizer = cv2.FaceRecognizerSF.create(
            model=FACE_RECOGNITION_MODEL, config=""
        )
        print("Models loaded successfully.")

        # --- THE FIX IS HERE ---
        # The script now looks inside the 'Data' folder for subfolders to process.
        base_image_folder = "Data"
        if not os.path.isdir(base_image_folder):
             print(f"Error: The main '{base_image_folder}' directory was not found.")
        else:
            # Loop through each subfolder in the 'Data' directory (e.g., 'Lalith')
            for person_name in os.listdir(base_image_folder):
                person_folder_path = os.path.join(base_image_folder, person_name)
                if os.path.isdir(person_folder_path):
                    # Call the main function to run the process for each person
                    generate_face_embeddings(person_folder_path, face_detector, face_recognizer)

    except Exception as e:
        print(f"An error occurred while loading the models or running the process: {e}")
else:
    print("\nSkipping execution because one or both models are not available.")

print("\nCell 4/4: Execution finished.")

