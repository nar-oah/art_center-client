import os
from psycopg.types.json import Json
from db import DatabaseManager, Feature
from analyzer import ImageAnalyzer
from pose.inference import PoseExtractorPipeline
from tqdm import tqdm


def main(directory: str) -> None:
    def get_images(directory: str) -> list[str]:
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
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
    import sys

    main(sys.argv[1])
