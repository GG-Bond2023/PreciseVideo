import numpy as np
import cv2
import os

target_colors = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85)
]


def extract_pose_from_image(image):

    h, w, _ = image.shape
    pose = []

    for color in target_colors:
        mask = cv2.inRange(image, np.array(color), np.array(color))
        ys, xs = np.where(mask > 0)

        if len(xs) > 0:
            u = int(np.mean(xs))
            v = int(np.mean(ys))
        else:
            u, v = -1, -1  
        pose.append([u, v])

    return np.array(pose)  # shape: (18, 2)


def load_pose_sequence_from_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".png") or f.endswith(".jpg")
    ])

    pose_sequence = []
    for file in image_files:
        img = cv2.imread(file)
        pose = extract_pose_from_image(img)
        pose_sequence.append(pose)

    return np.array(pose_sequence)  # shape: (m, 18, 2)


def compute_optimal_reference_frame(poses):
    m, l, _ = poses.shape
    d = np.zeros((m, m, l))

    for i in range(m):
        for j in range(m):
            dist = np.linalg.norm(poses[i] - poses[j], axis=1)
            dist[np.any((poses[i] == -1) | (poses[j] == -1), axis=1)] = 1e6 
            d[i, j] = np.exp(dist)

    total_displacement = d.sum(axis=(1, 2))
    alpha = np.argmin(total_displacement)
    return alpha

if __name__ == "__main__":
    folder_path = "/media/dlz/Data/workspace/my_work/VideoSD_v1_2/input/major_re/3-7/pose/8" 
    pose_sequence = load_pose_sequence_from_folder(folder_path)
    alpha = compute_optimal_reference_frame(pose_sequence)
    print("alpha =", alpha)
