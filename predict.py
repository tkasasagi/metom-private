from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

class MetomPredictor:
    def __init__(self, model_path=".", device="cuda", torch_dtype=torch.float32):
        """
        Initialize the Metom predictor
        
        Args:
            model_path: Path to the model directory (default: current directory)
            device: Device to run on ("cuda" or "cpu")
            torch_dtype: Torch data type (torch.float32, torch.float16, or torch.bfloat16)
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            _attn_implementation="eager",
            trust_remote_code=True
        ).to(device=device)
    
    def predict_image(self, image_path, k=1, return_probs=True):
        """
        Predict character from image
        
        Args:
            image_path: Path to the image file
            k: Number of top predictions to return (default: 1)
            return_probs: Whether to return probabilities (default: True)
            
        Returns:
            If k=1 and return_probs=True: tuple (label, probability)
            If k=1 and return_probs=False: string label
            If k>1 and return_probs=True: list of tuples [(label, prob), ...]
            If k>1 and return_probs=False: list of labels [label, ...]
        """
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        image_array = self.processor(images=image, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.torch_dtype
        )
        
        # Get predictions
        with torch.inference_mode():
            if k == 1:
                if return_probs:
                    results = self.model.get_topk_labels(image_array, k=1, return_probs=True)[0]
                    return results[0]  # Return single tuple (label, prob)
                else:
                    results = self.model.get_predictions(image_array)
                    return results[0]  # Return single label
            else:
                return self.model.get_topk_labels(image_array, k=k, return_probs=return_probs)[0]

def predict_image(image_path, k=1, return_probs=True, model_path=".", device="cuda", torch_dtype=torch.float32):
    """
    Simple function to predict character from image
    
    Args:
        image_path: Path to the image file
        k: Number of top predictions to return (default: 1)
        return_probs: Whether to return probabilities (default: True)
        model_path: Path to the model directory (default: current directory)
        device: Device to run on ("cuda" or "cpu")
        torch_dtype: Torch data type
        
    Returns:
        Prediction results (format depends on k and return_probs parameters)
    """
    predictor = MetomPredictor(model_path, device, torch_dtype)
    return predictor.predict_image(image_path, k, return_probs)

if __name__ == "__main__":
    # Example usage
    print("Testing prediction function...")
    
    # Single prediction with probability
    result = predict_image("examples/example1_4E00.jpg")
    print(f"Single prediction: {result}")
    
    # Top-3 predictions with probabilities
    results = predict_image("examples/example2_5B9A.jpg", k=3)
    print(f"Top-3 predictions: {results}")
    
    # Single prediction without probability
    result = predict_image("examples/example3_5009.jpg", return_probs=False)
    print(f"Single prediction (no prob): {result}")