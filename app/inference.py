import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image_for_tflite(image: Image.Image, input_shape: tuple, input_type: np.dtype):
    """
    Prepares a PIL image for TFLite model inference.
    
    Args:
        image (PIL.Image.Image): The uploaded image
        input_shape (tuple): Model expected shape, e.g., (1, 224, 224, 3)
        input_type (np.dtype): Model expected dtype, e.g., np.uint8 or np.float32

    Returns:
        np.ndarray: Preprocessed image ready for TFLite interpreter
    """
    # Resize to model input size
    target_height, target_width = input_shape[1], input_shape[2]
    image = image.resize((target_width, target_height))

    # Convert to numpy array
    image_np = np.array(image)

    # Convert to expected dtype
    if input_type == np.uint8:
        # If image values are 0-1 floats, scale to 0-255
        if image_np.dtype != np.uint8:
            if np.max(image_np) <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
    elif input_type == np.float32:
        image_np = image_np.astype(np.float32)

    # Add batch dimension if model expects 4D input
    if len(input_shape) == 4 and image_np.ndim == 3:
        image_np = np.expand_dims(image_np, axis=0)

    return image_np


class MaskDetector:
    def __init__(self, model_path: str):
        """
        Initializes the TFLite interpreter and input/output details.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image: Image.Image):
        """
        Preprocesses an image for the TFLite model using the correct dtype and shape.
        """
        return preprocess_image_for_tflite(
            image, self.input_details[0]['shape'], self.input_details[0]['dtype']
        )

    def predict(self, image: Image.Image):
        """
        Runs inference on a PIL image and returns the model output.
        """
        processed_image = self.preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
