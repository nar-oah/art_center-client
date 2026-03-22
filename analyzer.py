import torch
from PIL import Image
from typing import List, Optional, Tuple
from transformers import Siglip2Model, Siglip2Processor
from transformers.models.siglip2.modeling_siglip2 import Siglip2Output


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
        self.prompts = [f"A digital artwork showing {label}." for label in self.labels]

    def get_data(self, path: str, threshold: float) -> Tuple[List[float], List[str]]:
        def get_vector(tensor: Optional[torch.Tensor]) -> List[float]:
            return tensor[0].cpu().tolist() if tensor is not None else []

        def get_category(tensor: Optional[torch.Tensor]) -> List[str]:
            probs = get_vector(torch.sigmoid(tensor)) if tensor is not None else []
            return [self.labels[i] for i, prob in enumerate(probs) if prob > threshold]

        image = Image.open(path).convert("RGB")
        inputs = self.processor(
            text=self.prompts,
            images=image,
            padding="max_length",
            max_num_patches=256,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs: Siglip2Output = self.model(**inputs)
        return get_vector(outputs.image_embeds), get_category(outputs.logits_per_image)

    def clear_vram(self) -> None:
        del self.model
        del self.processor
        torch.cuda.empty_cache()
