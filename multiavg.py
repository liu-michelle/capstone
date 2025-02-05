import os
import cv2
import dlib
import glob
import math
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

predictor_model = 'shape_predictor_68_face_landmarks.dat'

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
    # Get list of image files (matches jpg, jpeg, png)
    image_paths = glob.glob(os.path.join(pics_folder, '*.[jp][pn]g'))
    if len(image_paths) == 0:
        raise ValueError("No images found in the folder: {}".format(pics_folder))
    
    images = []          # For processing (float32)
    original_images = [] # For display (8-bit)
    points_list = []
    
    # Load the first image to get a reference size.
    ref_img_uint8 = cv2.imread(image_paths[0])
    if ref_img_uint8 is None:
        raise FileNotFoundError("Could not load image: {}".format(image_paths[0]))
    h, w = ref_img_uint8.shape[:2]
    
    # Process each image: read, resize, detect landmarks on 8-bit image, then convert to float32.
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
        # Save the original 8-bit image for display.
        original_images.append(img_uint8)
        # Convert image to float32 for warping and blending.
        img = np.float32(img_uint8)
        images.append(img)
        points_list.append(pts)
    
    if len(images) == 0:
        raise ValueError("No valid face images found in folder: {}".format(pics_folder))
    
    # Compute the average landmark positions over all images.
    points_avg = np.mean(np.array(points_list), axis=0)
    
    # Compute Delaunay triangulation on the average points.
    triangles = get_triangles(points_avg)
    
    # Prepare an output image accumulator (using float32 for accumulation)
    img_accum = np.zeros((h, w, 3), dtype=np.float32)
    
    # Process each triangle: warp from each image to the average shape and blend.
    for tri_indices in triangles:
        # Average triangle vertices.
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

    # Convert the averaged face image to uint8 and save it.
    img_result = np.uint8(img_accum)
    cv2.imwrite(output_path, img_result)
    
    return img_result, original_images

def display_grid_and_merged(original_images, merged_face):
    """
    Displays a grid of the original face images alongside the merged face.
    The grid of original images appears on the left and the merged face is shown on the right.
    """
    # Convert images from BGR to RGB for display with matplotlib.
    original_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in original_images]
    merged_face_rgb = cv2.cvtColor(merged_face, cv2.COLOR_BGR2RGB)

    bilateral = cv2.bilateralFilter(merged_face_rgb, d=9, sigmaColor=75, sigmaSpace=75)

    # Detect edges using Canny
    edges = cv2.Canny(merged_face_rgb, threshold1=100, threshold2=200)

    # Convert edges to a 3-channel image so we can overlay them
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend the edges with the bilateral filtered image
    alpha = 0.9
    blended = cv2.addWeighted(bilateral, alpha, edges_colored, 1-alpha, 0)

    
    n = len(original_images_rgb)
    # Determine grid size (attempt a square-like grid)
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)
    
    # Create a figure with gridspec:
    # Left: grid of original faces.
    # Right: merged face.
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    
    # Left panel: grid of original images.
    ax_grid = plt.subplot(gs[0])
    ax_grid.axis('off')
    
    # Create a new figure for the grid of images.
    # Instead of placing them as subplots, we can combine them into one image.
    # We'll create a blank canvas and paste each face into it.
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

    # Display the grid of original images on the left panel.
    ax_grid.imshow(grid_image)
    
    # Right panel: merged (averaged) face.
    ax_merged = plt.subplot(gs[1])
    ax_merged.imshow(blended)
    ax_merged.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pics_folder = 'pics'
    output_file = 'merged_face.jpg'
    merged_face, original_images = average_faces_in_folder(pics_folder, output_file)
    print("Averaged face saved as:", output_file)
    
    display_grid_and_merged(original_images, merged_face)
