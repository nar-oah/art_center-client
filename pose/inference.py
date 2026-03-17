import cv2
import torch
import numpy as np
from ultralytics.models.yolo import YOLO
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional
from smplx import PureSMPLestX

YOLO_PATH = "models/yolo26n.pt"
SMPLX_PATH = "models/smplest_x_h.pth.tar"


class PoseExtractorPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(YOLO_PATH)
        self.smplx = PureSMPLestX().to(self.device)
        ckpt = torch.load(SMPLX_PATH, map_location=self.device)
        state_dict = ckpt["network"] if "network" in ckpt else ckpt
        clean_state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if "smplx_layer" not in k
        }
        self.smplx.load_state_dict(clean_state_dict, strict=False)
        self.smplx.eval()

    def _rot6d_to_axis_angle(self, rot6d_array: np.ndarray) -> np.ndarray:
        N = rot6d_array.shape[0]
        rot6d = rot6d_array.reshape(N, 2, 3)
        a1 = rot6d[:, 0, :]
        a2 = rot6d[:, 1, :]

        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        b2_unnorm = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2_unnorm / np.linalg.norm(b2_unnorm, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)
        rot_matrices = np.stack((b1, b2, b3), axis=-1)

        axis_angles = R.from_matrix(rot_matrices).as_rotvec()
        return axis_angles

    @torch.no_grad()
    def process_image_bytes(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        results = self.yolo(img, classes=[0], verbose=False)  # class 0 是人
        if len(results[0].boxes) == 0:
            return None

        # 获取最大的人体框并裁剪
        box = results[0].boxes[0].xyxy.cpu().numpy().squeeze().astype(int)
        x1, y1, x2, y2 = box
        cropped_img = img[y1:y2, x1:x2]

        # 3. 尺寸变换与归一化 (SMPLest-X 标准输入)
        resized_img = cv2.resize(cropped_img, (256, 256))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_img).float() / 255.0
        # 标准 ImageNet 归一化
        tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor_img = ((tensor_img - mean) / std).to(self.device)

        # 4. 推理获取 6D 参数
        raw_params = self.smplx(tensor_img)

        # 5. 转为 NumPy 并格式化为 Blender 轴角
        result_dict: Dict[str, np.ndarray] = {}
        for key in ["body_root_pose", "body_pose", "lhand_pose", "rhand_pose"]:
            rot6d_cpu = raw_params[key].cpu().numpy().reshape(-1, 6)
            axis_angle = self._rot6d_to_axis_angle(rot6d_cpu)
            result_dict[key] = axis_angle

        return result_dict


if __name__ == "__main__":
    pipeline = PoseExtractorPipeline()
    pose_data = pipeline.process_image_bytes("image/test.jpg")
    result = pose_data["body_pose"].shape if pose_data else "无"
    print("身体参数维度:", result)
