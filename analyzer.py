import torch
from PIL import Image
from typing import List, Tuple
from transformers.models.siglip2.modeling_siglip2 import Siglip2Model
from transformers.models.siglip2.processing_siglip2 import Siglip2Processor


class ImageAnalyzer:
    def __init__(self, model_name: str = "google/siglip2-base-patch16-naflex") -> None:
        self.model = Siglip2Model.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
        self.processor = Siglip2Processor.from_pretrained(model_name)
        self.labels = [
            "pose",
            "makeup",
            "clothing",
            "hairstyle",
            "background",
            "graphic design",
        ]
        self.prompts = [f"This is a photo of {label}." for label in self.labels]

    def get_data(self, path: str, threshold: float) -> Tuple[List[float], List[str]]:
        def get_vector(tensor: torch.Tensor) -> List[float]:
            return tensor[0].cpu().tolist()

        def get_category(tensor: torch.Tensor) -> List[str]:
            probs = get_vector(torch.sigmoid(tensor))
            return [self.labels[i] for i, prob in enumerate(probs) if prob > threshold]

        image = Image.open(path).convert("RGB")
        inputs = self.processor(text=self.prompts, images=image).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return get_vector(outputs.image_embeds), get_category(outputs.logits_per_image)

    def clear_vram(self) -> None:
        del self.model
        del self.processor
        torch.cuda.empty_cache()
