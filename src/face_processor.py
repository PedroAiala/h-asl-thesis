import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from utils.config import load_config, Config


class FaceProcessor:
    def __init__(self):
        """
        Initialize the FaceProcessor with a face detection model. Receives configuration parameters.
    
        """
        
        cfg: Config = load_config()

        dp_config = cfg.data_processing

        self.dp_config = dp_config

        self.target_image_size = tuple(dp_config.target_image_size)
        self.sharpness_threshold = dp_config.sharpness_threshold

        print(f"Starting the model {dp_config.detector_model_name}...")
        self.detector = FaceAnalysis(name=dp_config.detector_model_name, providers=['CPUExecutionProvider'])
        self.detector.prepare(
            ctx_id=dp_config.detector_ctx_id, 
            det_size=tuple(dp_config.detector_det_size)
        )
        
        print("Model loaded successfully.")

    def preprocess_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Executes the full preprocessing pipeline on a single image:

        Args:
            image_rgb (np.ndarray): The input image in RGB format (as read by scikit-learn).

        Returns:
            np.ndarray | None: An aligned and validated face image, or None if the image
                               fails the filters (no face, low sharpness, etc.).
        """
        try:
            img_bgr = self._convert_to_bgr(image_rgb)

            faces = self.detector.get(img_bgr)
            if not faces:
                return None  

            main_face = faces[0]
            aligned_face = self._align_face(img_bgr, main_face['kps'])

            if not self._is_sharp_enough(aligned_face):
                return None  

            return aligned_face

        except Exception as e:
            print(f"Error while preprocessing image: {e}")
            return None
    

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts the embedding from a PRE-PROCESSED face image.

        Args:
            aligned_face (np.ndarray): The image of an already aligned and cropped face.

        Returns:
            np.ndarray | None: The normalized embedding vector, or None if an error occurs.
        """
        try:
            faces = self.detector.get(aligned_face)
            
            if not faces:
                return None

            embedding = faces[0]['embedding']
            
            norm = np.linalg.norm(embedding)
            return embedding / norm

        except Exception as e:
            print(f"Erro inesperado durante a extração do embedding: {e}")
            return None


    def _convert_to_bgr(self, img: np.ndarray) -> np.ndarray:
        """Converts image to the BGR uint8 format."""
        return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


    def calculate_sharpness(self, face):
        """Calculates the sharpness of a face using the variance of the Laplacian."""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()


    def _align_face(self, img, landmarks):
        """Aligns the face using facial landmarks."""

        dst_points = np.array([
            [30.2946, 51.6963], [65.5318, 51.6963],  # Eyes
            [48.0252, 71.7366],                       # Nose
            [33.5493, 92.3655], [62.7299, 92.3655]    # Mouth
        ], dtype=np.float32)

        tform = cv2.estimateAffinePartial2D(landmarks, dst_points)[0]

        return cv2.warpAffine(img, tform, self.target_image_size)
    

    def _is_sharp_enough(self, face: np.ndarray) -> bool:
        """Checks if the face image is sharp enough based on the sharpness threshold."""
        sharpness = self.calculate_sharpness(face)
        return sharpness > self.sharpness_threshold
    
    
    def _show_images(self, original, processed, titles, n=6):
        """Utility function to display original and processed images."""

        plt.figure(figsize=(15, 5))
        for i in range(n):
            plt.subplot(2, n, i+1)
            plt.imshow(original[i])
            plt.title(f"Original\n{titles[i]}")
            plt.axis('off')

            plt.subplot(2, n, n+i+1)
            plt.imshow(cv2.cvtColor(processed[i], cv2.COLOR_BGR2RGB))
            plt.title(f"Processada\nNitidez: {self.calculate_sharpness(processed[i]):.1f}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    




