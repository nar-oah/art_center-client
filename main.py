import os
from typing import List
from psycopg.types.json import Json
from db import DatabaseManager, Feature
from analyzer import ImageAnalyzer
from tqdm import tqdm


def run(directory: str) -> None:
    def get_images(directory: str) -> List[str]:
        return [
            os.path.join(directory, path)
            for path in os.listdir(directory)
            if path.endswith((".png", ".jpg", ".jpeg"))
        ]

    def get_feature(path: str) -> Feature:
        return path, analyzer.get_data(path, 0.1)

    db = DatabaseManager()
    analyzer = ImageAnalyzer()
    images = get_images(directory)
    features = [get_feature(path) for path in tqdm(images, desc="获取特征", unit="张")]
    analyzer.clear_vram()
    # f = lambda path: Json(pose_model.extract(path))
    f = lambda path: Json(0.1)
    [db.add_data(feature, f) for feature in tqdm(features, desc="获取动作", unit="张")]
    # pose_model.clear_vram()
    db.close()


if __name__ == "__main__":
    IMAGE_FOLDER = "./test_images"
    run(IMAGE_FOLDER)
