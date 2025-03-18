import io
import os
import pickle

import gymnasium as gym
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from stable_baselines3.common.type_aliases import GymStepReturn


class DataCollectWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, output_dir: str) -> None:
        super().__init__(env)
        assert self.render_mode == "rgb_array", "render_mode must be rgb_array for data collection"

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.episode_id = 0
        self._init_episode_data()

    def _init_episode_data(self) -> None:
        self.episode_data = {
            "episode_id": self.episode_id,
            "serialized_frames": [],
            "actions": [],
        }

    def _compress_image(self, image_array: np.ndarray, format="JPEG", quality=85):
        """Compress image using PIL with JPEG compression"""
        img = Image.fromarray(image_array)
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=quality)
        return buffer.getvalue()

    def _dump_episode_data(self) -> None:
        """Save the episode data to a parquet file"""
        table = pa.Table.from_pandas(pd.DataFrame(self.episode_data))
        pq.write_table(table, os.path.join(self.output_dir, f"episode_{self.episode_id}.parquet"))

        self.episode_id += 1
        self._init_episode_data()

    def step(self, action: int) -> GymStepReturn:
        """Collect frame and action data during the episode"""
        frame = self.render()
        self.episode_data["serialized_frames"].append(pickle.dumps(frame))
        self.episode_data["actions"].append(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self._dump_episode_data()

        return observation, reward, terminated, truncated, info
