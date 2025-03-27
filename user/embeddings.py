import os
import numpy as np
from collections import defaultdict


def load_user_embeddings(users_dir: str) -> tuple[dict, dict]:
    user_embeddings = defaultdict()
    face_colors = defaultdict()
    for user_dir in os.listdir(users_dir):
        user_path = os.path.join(users_dir, user_dir)
        encodings = []
        for file in os.listdir(user_path):
            if file.endswith(".npy"):
                try:
                    data = np.load(os.path.join(user_path, file))
                    if data.ndim == 1:
                        encodings.append(data)
                    elif data.ndim == 2:
                        encodings.extend(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        if encodings:
            user_embeddings[user_dir] = encodings
            if user_dir not in face_colors:
                face_colors[user_dir] = tuple(np.random.randint(0, 255, size=3).tolist())

    return user_embeddings, face_colors
