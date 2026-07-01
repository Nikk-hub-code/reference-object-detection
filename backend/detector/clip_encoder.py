from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIPEncoder:

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32"
    ):

        """
        Load CLIP model and processor
        """

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = CLIPModel.from_pretrained(
            model_name
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            model_name
        )

    def encode_image(self, image_path):

        """
        Convert image into CLIP embedding
        """

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.get_image_features(
                **inputs
            )

        # Some transformers versions return object wrappers
        if hasattr(outputs, "pooler_output"):
            image_features = outputs.pooler_output
        else:
            image_features = outputs

        # Normalize embedding
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        return image_features