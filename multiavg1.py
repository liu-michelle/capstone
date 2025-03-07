import os
import cv2
import dlib
import glob
import math
import datetime
import numpy as np
from scipy.spatial import Delaunay

# Folder definitions (adjust paths as necessary)
base_dir = "/Users/liumichelle/capstone"
predictor_model = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")
pics_folder = os.path.join(base_dir, "pics")
avgs_folder = os.path.join(base_dir, "avgs")
grids_folder = os.path.join(base_dir, "grids")

# Ensure folders exist.
for folder in [pics_folder, avgs_folder, grids_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_points(image_uint8):
    """Detects facial landmarks for the given 8-bit image and appends eight boundary points."""
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    faces = face_detector(image_uint8, 1)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    detected_face = faces[0]
    pose_landmarks = face_pose_predictor(image_uint8, detected_face)
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])
    h, w = image_uint8.shape[:2]
    points.extend([
        [0, 0],
        [w // 2, 0],
        [w - 1, 0],
        [w - 1, h // 2],
        [w - 1, h - 1],
        [w // 2, h - 1],
        [0, h - 1],
        [0, h // 2]
    ])
    return np.array(points, np.float32)

def get_triangles(points):
    """Computes Delaunay triangulation for the given points."""
    return Delaunay(points).simplices

def affine_transform(src, src_tri, dst_tri, size):
    """Warps the triangular region from src defined by src_tri to dst_tri of the given size."""
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def get_latest_file(folder, pattern):
    """Returns the most recent file in folder matching pattern."""
    files = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime)
    return files[-1] if files else None

def average_last_and_new(pics_folder, avgs_folder):
    """
    Averages the new capture (the most recent file in pics) with the most recent merged image
    (if it exists) using weights: previous merged image 80% and new capture 20%.
    Saves the new merged image to avgs and returns it.
    """
    new_capture_path = get_latest_file(pics_folder, '*.[jp][pn]g')
    if not new_capture_path:
        raise ValueError("No captured images found in folder: " + pics_folder)
    new_img = cv2.imread(new_capture_path)
    if new_img is None:
        raise FileNotFoundError("Could not load new capture image: " + new_capture_path)
    
    latest_merged_path = get_latest_file(avgs_folder, 'merged_face_*.jpg')
    if latest_merged_path:
        prev_img = cv2.imread(latest_merged_path)
        if prev_img is None:
            print("Warning: Previous merged image load failed; using new capture only.")
            images = [new_img]
            weights = [1.0]
        else:
            images = [prev_img, new_img]
            weights = [0.9, 0.1]
    else:
        images = [new_img]
        weights = [1.0]
    
    # Resize all images to match the first image.
    h, w = images[0].shape[:2]
    images = [cv2.resize(img, (w, h)) for img in images]
    
    points_list = []
    for img in images:
        try:
            pts = get_points(img)
            points_list.append(pts)
        except Exception as e:
            print("Warning: skipping image due to error:", e)
    
    if not points_list:
        raise ValueError("No valid face images to process.")
    
    points_avg = np.mean(np.array(points_list), axis=0)
    triangles = get_triangles(points_avg)
    img_accum = np.zeros((h, w, 3), dtype=np.float32)
    
    for tri_indices in triangles:
        tri_avg = [points_avg[idx] for idx in tri_indices]
        rect_avg = cv2.boundingRect(np.float32([tri_avg]))
        x, y, rw, rh = rect_avg
        tri_avg_rect = [[pt[0] - x, pt[1] - y] for pt in tri_avg]
        mask = np.zeros((rh, rw, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_avg_rect), (1, 1, 1))
        img_patch = np.zeros((rh, rw, 3), dtype=np.float32)
        for (img, pts), weight in zip(zip(images, points_list), weights):
            tri_src = [pts[idx] for idx in tri_indices]
            rect_src = cv2.boundingRect(np.float32([tri_src]))
            x_src, y_src, rw_src, rh_src = rect_src
            tri_src_rect = [[pt[0] - x_src, pt[1] - y_src] for pt in tri_src]
            img_src_patch = img[y_src:y_src+rh_src, x_src:x_src+rw_src]
            warped_patch = affine_transform(img_src_patch, tri_src_rect, tri_avg_rect, (rw, rh))
            img_patch += weight * warped_patch
        roi = img_accum[y:y+rh, x:x+rw]
        roi = roi * (1 - mask) + img_patch * mask
        img_accum[y:y+rh, x:x+rw] = roi
    
    merged_face = np.uint8(img_accum)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = f"merged_face_{timestamp}.jpg"
    output_path = os.path.join(avgs_folder, merged_filename)
    cv2.imwrite(output_path, merged_face)
    print("New merged face saved as:", output_path)
    return merged_face

def load_all_captured_images(folder):
    """Loads all images from the given folder."""
    image_paths = sorted(glob.glob(os.path.join(folder, '*.[jp][pn]g')), key=os.path.getmtime)
    images = [cv2.imread(path) for path in image_paths if cv2.imread(path) is not None]
    return images

def create_grid_image(all_images, thumb_height=150):
    """
    Creates a grid image from all_images, preserving each image's aspect ratio.
    Each image is resized to a fixed height (thumb_height) and rows are padded
    to the maximum row width.
    """
    n = len(all_images)
    if n == 0:
        return None
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)
    thumbs = []
    for img in all_images:
        h, w = img.shape[:2]
        scale = thumb_height / float(h)
        new_w = int(round(w * scale))
        resized = cv2.resize(img, (new_w, thumb_height))
        thumbs.append(resized)
    row_wise_thumbs = []
    idx = 0
    for r in range(grid_rows):
        row = thumbs[idx : idx + grid_cols]
        idx += grid_cols
        if row:
            row_wise_thumbs.append(row)
    row_widths = [sum(t.shape[1] for t in row) for row in row_wise_thumbs]
    max_width = max(row_widths) if row_widths else 0
    row_images = []
    for row in row_wise_thumbs:
        row_height = max(t.shape[0] for t in row)
        row_img = np.full((row_height, max_width, 3), 255, dtype=np.uint8)
        x_offset = 0
        for t in row:
            h_t, w_t = t.shape[:2]
            row_img[0:h_t, x_offset:x_offset + w_t] = t
            x_offset += w_t
        row_images.append(row_img)
    grid_img = cv2.vconcat(row_images)
    return grid_img

def fit_to_canvas(img, canvas_width=1280, canvas_height=720):
    """
    Resizes img to fit within a canvas of canvas_width x canvas_height
    while preserving its aspect ratio, and centers it on a blank (white) canvas.
    """
    h, w = img.shape[:2]
    scale = min(canvas_width / w, canvas_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)
    x_offset = (canvas_width - new_w) // 2
    y_offset = (canvas_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def create_grid_file(all_captured_images, grids_folder):
    """Creates and saves a grid image of all captured images into the grids folder,
       then fits it to a 1280x720 canvas without warping the images."""
    grid_img = create_grid_image(all_captured_images, thumb_height=150)
    if grid_img is None:
        print("No images to create a grid.")
        return
    final_grid = fit_to_canvas(grid_img, canvas_width=1280, canvas_height=720)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_filename = f"grid_{timestamp}.jpg"
    grid_output_path = os.path.join(grids_folder, grid_filename)
    cv2.imwrite(grid_output_path, final_grid)
    print(f"Grid image saved as: {grid_output_path}")

def main():
    try:
        merged_face = average_last_and_new(pics_folder, avgs_folder)
    except Exception as e:
        print("Error during averaging:", e)
        return
    all_captured = load_all_captured_images(pics_folder)
    create_grid_file(all_captured, grids_folder)
    print("Done. Pictures are in 'pics', the averaged face is in 'avgs', and the grid is in 'grids'.")

# If running as a standalone script in TouchDesigner, call main().
main()
