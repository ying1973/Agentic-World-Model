import uuid
import base64
import tempfile
import hashlib
import os


class GlobalMediaCache:
    """
    全局缓存：管理图片与视频两种媒体。
    图片：media_id → bytes
    视频：video_id → { "path": ..., "frames": [...] }
    """

    def __init__(self):
        self.media_store = {}        # media_id -> bytes
        self.video_store = {}        # video_id -> dict(path, frames)

        self.image_hash_index = {}   # image_hash -> media_id
        self.materialized_files = {}

    def register(self, image_bytes: bytes):
        """
        注册图片（带去重）：
        - 如果 image_bytes 已注册过，直接返回已有 media_id 和 b64
        - 否则创建新的 media_id
        """
        # 1. 计算内容 hash
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        # 2. 已注册：直接返回
        if image_hash in self.image_hash_index:
            media_id = self.image_hash_index[image_hash]
            b64 = base64.b64encode(self.media_store[media_id]).decode()
            return media_id, b64

        # 3. 未注册：创建新条目
        media_id = "media_" + image_hash
        # media_id = "media_" + image_hash
        self.media_store[media_id] = image_bytes
        self.image_hash_index[image_hash] = media_id

        b64 = base64.b64encode(image_bytes).decode()
        return media_id, b64

    def get_bytes(self, media_id: str):
        return self.media_store.get(media_id)

    def materialize_to_temp_file(self, media_id: str):
        if media_id in self.materialized_files:
            rel_path = self.materialized_files[media_id]
            if os.path.exists(rel_path):
                return rel_path
            else:
                # 文件被外部删除，清理失效记录
                del self.materialized_files[media_id]

        data = self.get_bytes(media_id)
        if data is None:
            raise KeyError(f"[GlobalMediaCache] image_id '{media_id}' not found")

        rel_dir = "./output/temp_file"
        abs_dir = os.path.abspath(rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
            dir=abs_dir,
            prefix=f"{media_id}"
        )

        tmp.write(data)
        tmp.close()

        abs_path = tmp.name
        rel_path = "./" + os.path.relpath(abs_path, start=os.getcwd())

        self.materialized_files[media_id] = rel_path

        return rel_path



    def register_video(self, video_path: str):
        video_id = "video_" + uuid.uuid4().hex[:8]
        self.video_store[video_id] = {
            "path": video_path,
            "frames": []
        }
        return video_id

    def add_video_frames(self, video_id: str, frame_media_ids: list):
        if video_id not in self.video_store:
            raise KeyError(f"[GlobalMediaCache] video_id '{video_id}' not found")
        self.video_store[video_id]["frames"] = frame_media_ids

    def get_video_path(self, video_id: str):
        info = self.video_store.get(video_id)
        if info is None:
            raise KeyError(f"[GlobalMediaCache] video_id '{video_id}' not found")
        return info["path"]

    def get_video_frames(self, video_id: str):
        info = self.video_store.get(video_id)
        if info is None:
            raise KeyError(f"[GlobalMediaCache] video_id '{video_id}' not found")
        return info["frames"]


# 全局唯一实例
global_media_cache = GlobalMediaCache()
