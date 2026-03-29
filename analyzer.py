import torch
from torch import Tensor
from PIL import Image
from collections.abc import Callable
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
            "background element",
            "layout or composition",
            "color pairing",
            "clothing combination",
            "clothing silhouette",
            "clothing style or material",
            "hairstyle or hair ornament",
            "facial features or makeup",
            "pose or body gesture",
        ]
        self.prompts = [f"this image has a distinctive {l}." for l in self.labels]

    @torch.no_grad()
    def get_data(self, path: str, threshold: float) -> tuple[list[float], list[str]]:
        def get_vector(
            tensor: Tensor | None, f: Callable[[Tensor], Tensor] = lambda x: x
        ) -> list[float]:
            return f(tensor)[0].cpu().tolist() if isinstance(tensor, Tensor) else []

        def get_category(probs: list[float]) -> list[str]:
            category = [self.labels[i] for i, p in enumerate(probs) if p > threshold]
            return category if category else [self.labels[probs.index(max(probs))]]

        image = Image.open(path).convert("RGB")
        kwargs = {
            "padding": "max_length",
            "max_num_patches": 256,
            "return_tensors": "pt",
        }
        device = self.model.device
        inputs = self.processor(text=self.prompts, images=image, **kwargs).to(device)
        outputs: Siglip2Output = self.model(**inputs)
        probs = get_vector(outputs.logits_per_image, torch.sigmoid)
        return get_vector(outputs.image_embeds), get_category(probs)

    def clear_vram(self) -> None:
        del self.model
        del self.processor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    analyzer = ImageAnalyzer()
    analyzer.get_data(sys.argv[1], 0.1)
