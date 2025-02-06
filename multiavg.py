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

def average_faces_in_folder(pics_folder, output_path='merged_face.jpg'):
    """
    Averages all face images in the specified folder and returns the averaged image
    along with a list of original images (for display).
    """
    image_paths = glob.glob(os.path.join(pics_folder, '*.[jp][pn]g'))
    if len(image_paths) == 0:
        raise ValueError("No images found in the folder: {}".format(pics_folder))
    
    images = []          # For processing (float32)
    original_images = [] # For display (8-bit)
    points_list = []
    
    ref_img_uint8 = cv2.imread(image_paths[0])
    if ref_img_uint8 is None:
        raise FileNotFoundError("Could not load image: {}".format(image_paths[0]))
    h, w = ref_img_uint8.shape[:2]
    
    for path in image_paths:
        img_uint8 = cv2.imread(path)
        if img_uint8 is None:
            print("Warning: could not load image:", path)
            continue
        img_uint8 = cv2.resize(img_uint8, (w, h))
        try:
            pts = get_points(img_uint8)
        except Exception as e:
            print("Warning: skipping image {}: {}".format(path, e))
            continue
        original_images.append(img_uint8)
        img = np.float32(img_uint8)
        images.append(img)
        points_list.append(pts)
    
    if len(images) == 0:
        raise ValueError("No valid face images found in folder: {}".format(pics_folder))
    
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
        for img, pts in zip(images, points_list):
            tri_src = [pts[idx] for idx in tri_indices]
            rect_src = cv2.boundingRect(np.float32([tri_src]))
            x_src, y_src, rw_src, rh_src = rect_src
            tri_src_rect = [[pt[0] - x_src, pt[1] - y_src] for pt in tri_src]
            
            img_src_patch = img[y_src:y_src+rh_src, x_src:x_src+rw_src]
            warped_patch = affine_transform(img_src_patch, tri_src_rect, tri_avg_rect, (rw, rh))
            img_patch += warped_patch
        
        img_patch /= len(images)
        roi = img_accum[y:y+rh, x:x+rw]
        roi = roi * (1 - mask) + img_patch * mask
        img_accum[y:y+rh, x:x+rw] = roi

    img_result = np.uint8(img_accum)
    cv2.imwrite(output_path, img_result)
    
    return img_result, original_images

def display_grid_and_merged(original_images, merged_face):
    """
    Displays a grid of the original face images (small thumbnails) alongside the merged face.
    The grid of original images appears on the left and the merged face is shown on the right.
    """
    # Convert images from BGR to RGB for display.
    original_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in original_images]
    merged_face_rgb = cv2.cvtColor(merged_face, cv2.COLOR_BGR2RGB)

    bilateral = cv2.bilateralFilter(merged_face_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(merged_face_rgb, threshold1=100, threshold2=200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    alpha = 0.9
    blended = cv2.addWeighted(bilateral, alpha, edges_colored, 1 - alpha, 0)

    n = len(original_images_rgb)
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)

    # Adjust figure size and gridspec so that the merged image takes up more space.
    fig = plt.figure(figsize=(16, 8), linewidth=4, edgecolor='black')
    fig.patch.set_facecolor('#f2f2f2')
    # Left panel for grid thumbnails is smaller (width ratio 1) and merged face is larger (width ratio 2)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    
    # Left panel: grid of original images (thumbnails)
    ax_grid = plt.subplot(gs[0])
    ax_grid.axis('off')
    
    # Create a blank canvas for the grid of thumbnails.
    thumb_h, thumb_w = original_images_rgb[0].shape[0:2]
    grid_image = np.ones((grid_rows * thumb_h, grid_cols * thumb_w, 3), dtype=np.uint8) * 255

    for idx, face in enumerate(original_images_rgb):
        row = idx // grid_cols
        col = idx % grid_cols
        y0 = row * thumb_h
        y1 = y0 + thumb_h
        x0 = col * thumb_w
        x1 = x0 + thumb_w
        grid_image[y0:y1, x0:x1, :] = face

    ax_grid.imshow(grid_image)
    
    # Right panel: merged (averaged) face (larger display)
    ax_merged = plt.subplot(gs[1])
    ax_merged.imshow(blended)
    ax_merged.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    capture_from_webcam(save_folder=pics_folder)
    output_file = 'merged_face.jpg'
    merged_face, original_images = average_faces_in_folder(pics_folder, output_file)
    print("Averaged face saved as:", output_file)
    display_grid_and_merged(original_images, merged_face)
