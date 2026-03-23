import psycopg
from psycopg.types.json import Json
from collections.abc import Callable

type Feature = tuple[str, tuple[list[float], list[str]]]


class DatabaseManager:
    def __init__(self) -> None:
        self.conn = psycopg.connect()
        self.cursor = self.conn.cursor()

    def add_data(self, feature: Feature, get_pose: Callable[[str], Json]) -> None:
        path, (vector, categories) = feature
        pose = get_pose(path) if "pose" in categories else None
        insert_query = """
                INSERT INTO art_center (path, vector, categories, pose)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (path) DO UPDATE 
                SET vector = EXCLUDED.vector,
                    categories = EXCLUDED.categories,
                    pose = EXCLUDED.pose;
            """
        self.cursor.execute(insert_query, (path, vector, categories, pose))
        self.conn.commit()

    def close(self) -> None:
        self.cursor.close()
        self.conn.close()
