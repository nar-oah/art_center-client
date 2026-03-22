import os
import sys
from typing import List
from psycopg.types.json import Json
from db import DatabaseManager, Feature
from analyzer import ImageAnalyzer
from pose.inference import PoseExtractorPipeline
from tqdm import tqdm


def main(directory: str) -> None:
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
    pose = PoseExtractorPipeline()
    images = get_images(directory)
    features = [get_feature(path) for path in tqdm(images, desc="获取特征", unit="张")]
    analyzer.clear_vram()
    f = lambda path: Json(pose.get_pose(path))
    [db.add_data(feature, f) for feature in tqdm(features, desc="获取动作", unit="张")]
    pose.clear_vram()
    db.close()


if __name__ == "__main__":
    main(sys.argv[1])
