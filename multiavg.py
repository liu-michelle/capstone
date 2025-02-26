import os
import cv2
import dlib
import glob
import math
import datetime
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

predictor_model = 'shape_predictor_68_face_landmarks.dat'
pics_folder = 'pics'
avgs_folder = 'avgs'

# Ensure both folders exist.
for folder in [pics_folder, avgs_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def capture_from_webcam(save_folder=pics_folder):
    """
    Opens the webcam and waits for the user to press spacebar.
    When spacebar is pressed, the current frame is saved into save_folder.
    Press ESC to exit without capturing.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return

    print("Press Spacebar to capture an image, or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Webcam - Press Spacebar to Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("Exiting without capturing.")
            break
        elif key == 32:  # Spacebar key
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"captured_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Image saved to {filename}")
            break

    cap.release()
    cv2.destroyAllWindows()

def get_points(image_uint8):
    """
    Detects facial landmarks for the given 8-bit image and appends eight boundary points.
    """
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
    
    # Add eight boundary points (corners and midpoints)
    h, w = image_uint8.shape[:2]
    points.append([0, 0])
    points.append([w // 2, 0])
    points.append([w - 1, 0])
    points.append([w - 1, h // 2])
    points.append([w - 1, h - 1])
    points.append([w // 2, h - 1])
    points.append([0, h - 1])
    points.append([0, h // 2])
    
    return np.array(points, np.float32)

def get_triangles(points):
    """Computes Delaunay triangulation for the given points."""
    return Delaunay(points).simplices

def affine_transform(src, src_tri, dst_tri, size):
    """
    Warps the triangular region from the source image (src) given by src_tri to 
    the destination triangle (dst_tri) of the given size.
    """
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def get_latest_merged(avgs_folder):
    """
    Returns the path of the latest merged image from the avgs folder, if any.
    """
    merged_paths = sorted(glob.glob(os.path.join(avgs_folder, 'merged_face_*.jpg')), key=os.path.getmtime)
    return merged_paths[-1] if merged_paths else None

def average_last_and_new(pics_folder, avgs_folder):
    """
    Averages the new capture with the most recent merged image (if it exists)
    to produce a new merged face image. A new file is saved in the avgs folder
    with a unique timestamp.
    
    Returns the merged face image and the average landmark positions.
    """
    # Get the newest capture from pics_folder.
    capture_paths = sorted(glob.glob(os.path.join(pics_folder, '*.[jp][pn]g')), key=os.path.getmtime)
    if not capture_paths:
        raise ValueError("No captured images found in folder: {}".format(pics_folder))
    new_capture_path = capture_paths[-1]
    new_img = cv2.imread(new_capture_path)
    if new_img is None:
        raise FileNotFoundError("Could not load new capture image: {}".format(new_capture_path))
    
    # Check for an existing merged image.
    latest_merged_path = get_latest_merged(avgs_folder)
    if latest_merged_path:
        prev_img = cv2.imread(latest_merged_path)
        if prev_img is None:
            print("Warning: Could not load previous merged image. Using new capture only.")
            images = [new_img]
        else:
            images = [prev_img, new_img]
    else:
        images = [new_img]
    
    # Resize all images to the same dimensions based on the first image.
    h, w = images[0].shape[:2]
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (w, h))
    
    points_list = []
    for img in images:
        try:
            pts = get_points(img)
            points_list.append(pts)
        except Exception as e:
            print("Warning: skipping an image due to error:", e)
    
    if not points_list:
        raise ValueError("No valid face images to process.")
    
    # Compute the average landmark positions.
    points_avg = np.mean(np.array(points_list), axis=0)
    
    # Compute Delaunay triangulation on the average points.
    triangles = get_triangles(points_avg)
    img_accum = np.zeros((h, w, 3), dtype=np.float32)
    
    for tri_indices in triangles:
        tri_avg = [points_avg[idx] for idx in tri_indices]
        rect_avg = cv2.boundingRect(np.float32([tri_avg]))
        x, y, rw, rh = rect_avg
        
        tri_avg_rect = [[pt[0]-x, pt[1]-y] for pt in tri_avg]
        mask = np.zeros((rh, rw, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_avg_rect), (1, 1, 1))
        
        img_patch = np.zeros((rh, rw, 3), dtype=np.float32)
        for img, pts in zip(images, points_list):
            tri_src = [pts[idx] for idx in tri_indices]
            rect_src = cv2.boundingRect(np.float32([tri_src]))
            x_src, y_src, rw_src, rh_src = rect_src
            tri_src_rect = [[pt[0]-x_src, pt[1]-y_src] for pt in tri_src]
            img_src_patch = img[y_src:y_src+rh_src, x_src:x_src+rw_src]
            warped_patch = affine_transform(img_src_patch, tri_src_rect, tri_avg_rect, (rw, rh))
            img_patch += warped_patch
        
        img_patch /= len(images)
        roi = img_accum[y:y+rh, x:x+rw]
        roi = roi*(1-mask) + img_patch*mask
        img_accum[y:y+rh, x:x+rw] = roi

    merged_face = np.uint8(img_accum)
    # Create a new unique filename for the merged image.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(avgs_folder, f"merged_face_{timestamp}.jpg")
    cv2.imwrite(output_path, merged_face)
    print("New merged face saved as:", output_path)
    
    return merged_face, points_avg

def load_all_captured_images(folder):
    """
    Loads all images from the given folder.
    """
    image_paths = sorted(glob.glob(os.path.join(folder, '*.[jp][pn]g')), key=os.path.getmtime)
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images

def overlay_landmarks_and_lines_on_image(image, points, dot_color=(255, 0, 0),
                                         line_color=(0, 0, 255), dot_radius=4, thickness=2):
    """
    Returns a copy of the image with facial landmark dots and connecting lines.
    Assumes that the first 68 points in 'points' correspond to the dlib 68 landmarks.
    """
    overlay = image.copy()
    # Draw dots.
    for (x, y) in points.astype(int):
        cv2.circle(overlay, (x, y), dot_radius, dot_color, -1)
    
    # Define facial feature connections.
    pts = points[:68]
    connections = [
        list(range(0, 17)),          # Jawline
        list(range(17, 22)),         # Left eyebrow
        list(range(22, 27)),         # Right eyebrow
        list(range(27, 31)),         # Nose bridge
        list(range(30, 36)),         # Lower nose
        list(range(36, 42)) + [36],   # Left eye
        list(range(42, 48)) + [42],   # Right eye
        list(range(48, 60)) + [48],   # Outer lip
        list(range(60, 68)) + [60]    # Inner lip
    ]
    for connection in connections:
        for i in range(len(connection) - 1):
            pt1 = tuple(pts[connection[i]].astype(int))
            pt2 = tuple(pts[connection[i+1]].astype(int))
            cv2.line(overlay, pt1, pt2, line_color, thickness)
    return overlay

def display_grid_and_merged(merged_face, all_captured_images, points_avg):
    """
    Displays a grid of all captured images on the left and the merged face on the right.
    Every 5 seconds, the merged face display will switch to an overlay view (with facial
    landmark dots and connecting lines) for 2 seconds, then revert to the normal view.
    """
    # Convert images to RGB for matplotlib.
    original_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_captured_images]
    merged_rgb = cv2.cvtColor(merged_face, cv2.COLOR_BGR2RGB)

    bilateral = cv2.bilateralFilter(merged_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(merged_rgb, threshold1=100, threshold2=200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(bilateral, 0.9, edges_colored, 0.1, 0)

    # Create an overlay version with landmarks and lines.
    merged_overlay = overlay_landmarks_and_lines_on_image(merged_rgb, points_avg,
                                                          dot_color=(255, 255, 255),
                                                          line_color=(255, 255, 255),
                                                          dot_radius=4, thickness=2)

    n = len(original_rgb)
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)

    fig = plt.figure(figsize=(16, 8), linewidth=4, edgecolor='black')
    fig.patch.set_facecolor('#f2f2f2')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    
    # Left panel: grid of all captured thumbnails.
    ax_grid = plt.subplot(gs[0])
    ax_grid.axis('off')
    thumb_h, thumb_w = original_rgb[0].shape[0:2]
    grid_image = np.ones((grid_rows * thumb_h, grid_cols * thumb_w, 3), dtype=np.uint8) * 255
    for idx, face in enumerate(original_rgb):
        row = idx // grid_cols
        col = idx % grid_cols
        y0 = row * thumb_h
        y1 = y0 + thumb_h
        x0 = col * thumb_w
        x1 = x0 + thumb_w
        grid_image[y0:y1, x0:x1, :] = face
    ax_grid.imshow(grid_image)
    
    # Right panel: merged face.
    ax_merged = plt.subplot(gs[1])
    ax_merged.axis('off')
    im_disp = ax_merged.imshow(blended)
    
    # Timer callback to alternate overlay.
    state = {"current": "normal"}
    merged_data = {"normal": blended, "overlay": merged_overlay}
    
    def toggle_overlay():
        if state["current"] == "normal":
            im_disp.set_data(merged_data["overlay"])
            state["current"] = "overlay"
            timer.interval = 2000
        else:
            im_disp.set_data(merged_data["normal"])
            state["current"] = "normal"
            timer.interval = 5000
        fig.canvas.draw_idle()
        timer.start()

    timer = fig.canvas.new_timer(interval=5000)
    timer.add_callback(toggle_overlay)
    timer.start()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Capture a new image (if desired).
    capture_from_webcam(save_folder=pics_folder)
    
    # Average the new capture with the last merged face (if it exists)
    # and save the new merged image as a new file.
    merged_face, points_avg = average_last_and_new(pics_folder, avgs_folder)
    
    # Load all captured images from the pics folder for grid display.
    all_captured = load_all_captured_images(pics_folder)
    
    # Display the grid (all captures) and the merged face with periodic overlay switching.
    display_grid_and_merged(merged_face, all_captured, points_avg)
