import os
import pickle
from argparse import ArgumentParser

import imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Load a font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()


def main():
    parser = ArgumentParser()
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet_path)

    # Deserialize the frames from the 'serialized_frames' column
    frames = df["serialized_frames"].apply(pickle.loads).tolist()
    actions = df["actions"].tolist()

    # Iterate through frames and add action number
    for i, (frame, action) in enumerate(zip(frames, actions)):
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        text = str(action)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        width, height = pil_image.size
        position = (width - text_width - 5, height - text_height - 5)
        draw.text(position, text, font=font, fill="black")
        frames[i] = np.array(pil_image)

    # Save the frames as an mp4 with 60fps
    imageio.mimsave(args.save_path, frames, format="mp4", fps=60)


if __name__ == "__main__":
    main()
