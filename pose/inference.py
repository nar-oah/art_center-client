import cv2
from numpy.typing import NDArray
import torch
import numpy as np
import ultralytics.engine.results as results
from ultralytics.models.yolo import YOLO
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, Mapping, Optional, Tuple, List
from pose.smplx import PureSMPLestX
from pathlib import Path

DIR = Path(__file__).resolve().parent
YOLO_PATH = DIR / "models" / "yolo26n.pt"
SMPLX_PATH = DIR / "models" / "smplest_x_h.pth.tar"


class PoseExtractorPipeline:
    def __init__(self) -> None:
        def get_state_dict(device: torch.device) -> Mapping[str, Any]:
            ckpt = torch.load(SMPLX_PATH, map_location=device)
            state_dict = ckpt["network"] if "network" in ckpt else ckpt
            return {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
                if "smplx_layer" not in k
            }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(YOLO_PATH)
        self.smplx = PureSMPLestX().to(self.device)
        self.smplx.load_state_dict(get_state_dict(self.device), strict=False)
        self.smplx.eval()

    def get_angle(self, rot6d: np.ndarray) -> np.ndarray:
        def get_vector(rot6d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            N = rot6d.shape[0]
            rot6d = rot6d.reshape(N, 2, 3)
            return rot6d[:, 0, :], rot6d[:, 1, :]

        def get_matrix(a1: np.ndarray, a2: np.ndarray) -> NDArray:
            b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
            b2_unnorm = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
            b2 = b2_unnorm / np.linalg.norm(b2_unnorm, axis=-1, keepdims=True)
            b3 = np.cross(b1, b2)
            return np.stack((b1, b2, b3), axis=-1)

        axis_angles = R.from_matrix(get_matrix(*get_vector(rot6d))).as_rotvec()
        return axis_angles

    @torch.no_grad()
    def get_pose(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        def get_human(boxes: results.Boxes, img: np.ndarray) -> np.ndarray:
            x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy().squeeze().astype(int)
            return img[y1:y2, x1:x2]

        def get_tensor(img: np.ndarray) -> torch.Tensor:
            resized_img = cv2.resize(img, (192, 256))
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb_img).float() / 255.0

        def mod_tensor(tensor: torch.Tensor) -> torch.Tensor:
            tensor_img = tensor.permute(2, 0, 1).unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            return ((tensor_img - mean) / std).to(self.device)

        def get_angle(params: Dict[str, torch.Tensor], key: str) -> np.ndarray:
            rot6d_cpu = params[key].cpu().numpy().reshape(-1, 6)
            return self.get_angle(rot6d_cpu)

        results: List[results.Results] = self.yolo(path, classes=[0], verbose=False)
        if boxes := results[0].boxes:
            img = get_human(boxes, results[0].orig_img)
            params = self.smplx(mod_tensor(get_tensor(img)))
            keys = ["body_root_pose", "body_pose", "lhand_pose", "rhand_pose"]
            return {key: get_angle(params, key) for key in keys}

    def clear_vram(self) -> None:
        del self.yolo
        del self.smplx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pipeline = PoseExtractorPipeline()
    path = DIR / "image" / "test.jpg"
    pose_data = pipeline.get_pose(str(path))
    print("根参数:", pose_data['body_root_pose'])
    print("身体参数:", pose_data['body_pose'])
    print("左手参数:", pose_data['lhand_pose'])
    print("右手参数:", pose_data['rhand_pose'])
