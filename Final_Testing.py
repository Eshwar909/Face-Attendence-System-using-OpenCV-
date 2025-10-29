# =============================================================================
# CELL 1: Imports and Initial Setup
# =============================================================================
# Purpose: Import all necessary libraries and configure the environment.
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import warnings
import uuid
from datetime import datetime  # <-- ADDED for timestamping

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore")
print("Cell 1/4: Libraries imported and environment configured.")


# =============================================================================
# CELL 2: Model and Known Embeddings Loader
# =============================================================================
# Purpose: Verify model files exist and load all previously generated embeddings
#          from the 'embeddings' directory into memory.
# -----------------------------------------------------------------------------
# --- Model Information ---
FACE_DETECTION_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx"

def check_model_exists(model_name):
    """Checks if a model file exists in the current directory."""
    if not os.path.exists(model_name):
        print(f"Error: Model file not found: '{model_name}'.")
        return False
    print(f"Model file '{model_name}' found.")
    return True

def load_known_faces(embeddings_base_folder):
    """Loads all saved embeddings and their corresponding labels."""
    known_embeddings = []
    known_labels = []
    if not os.path.isdir(embeddings_base_folder):
        print(f"Warning: Embeddings directory not found at '{embeddings_base_folder}'.")
        return [], []

    for person_name in os.listdir(embeddings_base_folder):
        person_dir = os.path.join(embeddings_base_folder, person_name)
        if os.path.isdir(person_dir):
            embeddings_path = os.path.join(person_dir, f"{person_name}_embeddings.npy")
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path)
                labels = [person_name] * len(embeddings)
                known_embeddings.extend(embeddings)
                known_labels.extend(labels)
                print(f"- Loaded {len(embeddings)} embeddings for '{person_name}'.")

    return known_embeddings, known_labels

# --- Execute checks and loaders ---
detector_ready = check_model_exists(FACE_DETECTION_MODEL)
recognizer_ready = check_model_exists(FACE_RECOGNITION_MODEL)

known_face_embeddings, known_face_labels = [], []
if detector_ready and recognizer_ready:
    print("\nLoading known face embeddings...")
    known_face_embeddings, known_face_labels = load_known_faces("embeddings")
    if not known_face_embeddings:
        print("Could not find any pre-saved embeddings to compare against.")

print("Cell 2/4: Loaders finished.")


# =============================================================================
# CELL 3: Recognition Function
# =============================================================================
# Purpose: Define the function that captures a single image, generates an
#          embedding, and compares it against the known dataset.
# -----------------------------------------------------------------------------

# --- NEW: Helper function to save failed images ---
def save_false_case(image_frame, reason):
    """Saves an image that failed recognition to the 'False_Cases' folder."""
    try:
        false_cases_folder = "False_Cases"
        os.makedirs(false_cases_folder, exist_ok=True)

        # Generate a timestamp (e.g., "20251024_115501")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a unique filename
        false_filename = f"false_case_{reason}_{timestamp}.jpg"
        false_image_path = os.path.join(false_cases_folder, false_filename)

        # Save the image
        cv2.imwrite(false_image_path, image_frame)
        print(f"  - Saved unrecognized image to '{false_image_path}'")
    except Exception as e:
        print(f"  - Warning: Could not save false case image. Error: {e}")


def recognize_face_from_capture(detector, recognizer, known_embeddings, known_labels):
    """
    Captures a single image, generates a face embedding, compares it against
    the known database, and prints the result.
    """
    # --- Matching Threshold ---
    COSINE_THRESHOLD = 0.40

    # --- Capture a single frame ---
    print("\nOpening webcam for capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture frame from webcam.")
        return

    print("Image captured successfully. Processing for recognition...")

    # --- Detect Face ---
    h, w, _ = frame.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        print("\n--- RESULT: Face Not Matched (Reason: No face detected in the image) ---")
        save_false_case(frame, "NoFaceDetected")  # <-- ADDED
        return

    # --- Generate Embedding for the detected face ---
    try:
        face_coords = faces[0]
        aligned_face = recognizer.alignCrop(frame, face_coords)
        new_embedding = recognizer.feature(aligned_face)

        best_match_name = "Unknown"
        best_match_score = 0

        # --- Compare with Known Faces ---
        for i, known_embedding in enumerate(known_embeddings):
            score = recognizer.match(new_embedding, known_embedding, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_match_score:
                best_match_score = score
                best_match_name = known_labels[i]

        # --- Determine and Print the Final Result ---
        if best_match_score >= COSINE_THRESHOLD:
            print(f"\n--- RESULT: Face Matched: {best_match_name} (Confidence Score: {best_match_score:.2f}) ---")

            try:
                # Define the main data folder and the user-specific subfolder for saving the image.
                base_data_folder = "Data"
                image_dir = os.path.join(base_data_folder, best_match_name)
                output_dir = os.path.join("embeddings", best_match_name)
                os.makedirs(image_dir, exist_ok=True)

                # Save the captured image with a unique name in the correct path.
                img_name = os.path.join(image_dir, f"{best_match_name}_{uuid.uuid1()}.jpg")
                cv2.imwrite(img_name, frame)
                print(f"  - Saved new reference image to '{img_name}'")

                # Load existing data, append new embedding, and save back
                embeddings_path = os.path.join(output_dir, f"{best_match_name}_embeddings.npy")
                labels_path = os.path.join(output_dir, f"{best_match_name}_labels.npy")

                if os.path.exists(embeddings_path):
                    person_embeddings = list(np.load(embeddings_path))
                    person_labels = list(np.load(labels_path))
                else:
                    person_embeddings, person_labels = [], []

                person_embeddings.append(new_embedding)
                person_labels.append(best_match_name)

                np.save(embeddings_path, np.array(person_embeddings))
                np.save(labels_path, np.array(person_labels))

                print(f"  - Updated embeddings for '{best_match_name}'. Total is now {len(person_embeddings)}.")
            except Exception as e:
                print(f"  - Warning: Could not save new data. Error: {e}")

        else:
            print(f"\n--- RESULT: Face Not Matched (Closest match was {best_match_name}, but score {best_match_score:.2f} was below threshold {COSINE_THRESHOLD}) ---")
            save_false_case(frame, f"FailedMatch_{best_match_name}")  # <-- ADDED

    except Exception as e:
        print(f"An error occurred during recognition: {e}")

print("Cell 3/4: Recognition function defined.")


# =============================================================================
# CELL 4: Execution Block
# =============================================================================
# Purpose: Load the models and call the main recognition function.
# -----------------------------------------------------------------------------
if detector_ready and recognizer_ready: # <-- MODIFIED (Removed check for known_face_embeddings)
    print("\nLoading models for recognition...")
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

        # Check if embeddings are available *after* loading models
        if not known_face_embeddings:
            print("\nWarning: No pre-saved embeddings were found.")
            print("Running in 'detection only' mode. All faces will be logged as 'Unknown'.")

        # Run the recognition process
        recognize_face_from_capture(face_detector, face_recognizer, known_face_embeddings, known_face_labels)

    except Exception as e:
        print(f"A critical error occurred: {e}")
else:
    print("\nSkipping execution due to missing models.") # <-- MODIFIED

print("\nCell 4/4: Execution finished.")

