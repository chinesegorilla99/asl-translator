import numpy as np


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    if len(landmarks) != 63:
        raise ValueError(f"Expected 63 features, got {len(landmarks)}")
    
    coords = landmarks.reshape(21, 3)
    
    wrist = coords[0].copy()
    centered = coords - wrist
    
    index_mcp = centered[5]
    pinky_mcp = centered[17]
    palm_width = np.linalg.norm(index_mcp - pinky_mcp)
    
    if palm_width > 0.001:
        normalized = centered / palm_width
    else:
        normalized = centered
    
    palm_vector = index_mcp - pinky_mcp
    palm_vector_2d = palm_vector[:2]
    palm_norm = np.linalg.norm(palm_vector_2d)
    if palm_norm > 0.001:
        angle = np.arctan2(palm_vector_2d[1], palm_vector_2d[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normalized[:, :2] = normalized[:, :2] @ rotation.T
    
    return normalized.flatten().astype(np.float32)


def normalize_batch(features: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(features)
    for i in range(len(features)):
        normalized[i] = normalize_landmarks(features[i])
    return normalized


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    if len(landmarks) != 63:
        raise ValueError(f"Expected 63 features, got {len(landmarks)}")
    
    coords = landmarks.reshape(21, 3)
    mirrored = coords.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    
    return mirrored.flatten().astype(np.float32)
