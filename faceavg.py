import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt

predictor_model = 'shape_predictor_68_face_landmarks.dat'

def face_average(img1_path, img2_path, alpha=0.5, output_path='merge.jpg'):
    """
    Averages two face images by morphing their facial landmarks.
    
    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        alpha (float): Blending ratio between the two images (default=0.5).
        output_path (str): Path to save the resulting morphed image (default='merge.jpg').
    
    Returns:
        np.ndarray: The resulting morphed image.
    """
    def get_points(image):
        """Detect face landmarks and add boundary points."""
        face_detector = dlib.get_frontal_face_detector()
        face_pose_predictor = dlib.shape_predictor(predictor_model)
        try:
            detected_face = face_detector(image, 1)[0]
        except:
            raise ValueError('No face detected in the image.')
        
        pose_landmarks = face_pose_predictor(image, detected_face)
        points = []
        for p in pose_landmarks.parts():
            points.append([p.x, p.y])

        x = image.shape[1] - 1
        y = image.shape[0] - 1
        points.append([0, 0])
        points.append([x // 2, 0])
        points.append([x, 0])
        points.append([x, y // 2])
        points.append([x, y])
        points.append([x // 2, y])
        points.append([0, y])
        points.append([0, y // 2])

        return np.array(points)

    def get_triangles(points):
        """Perform Delaunay triangulation on the points."""
        return Delaunay(points).simplices

    def affine_transform(input_image, input_triangle, output_triangle, size):
        """Apply affine transformation to warp a triangle."""
        warp_matrix = cv2.getAffineTransform(
            np.float32(input_triangle), np.float32(output_triangle))
        output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return output_image

    # Read and resize images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both input images not found.")
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert images to float32
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Get facial landmarks
    points1 = get_points(img1)
    points2 = get_points(img2)

    # Compute average points
    points_avg = (1 - alpha) * np.array(points1) + alpha * np.array(points2)

    # Prepare output image
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    # Perform Delaunay triangulation
    triangles = get_triangles(points_avg)

    # Process each triangle
    for tri_indices in triangles:
        x, y, z = tri_indices

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri_avg = [points_avg[x], points_avg[y], points_avg[z]]

        rect1 = cv2.boundingRect(np.float32([tri1]))
        rect2 = cv2.boundingRect(np.float32([tri2]))
        rect_avg = cv2.boundingRect(np.float32([tri_avg]))

        tri1_rect = [(p[0] - rect1[0], p[1] - rect1[1]) for p in tri1]
        tri2_rect = [(p[0] - rect2[0], p[1] - rect2[1]) for p in tri2]
        tri_avg_rect = [(p[0] - rect_avg[0], p[1] - rect_avg[1]) for p in tri_avg]

        img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
        img2_rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

        size = (rect_avg[2], rect_avg[3])
        warped_img1 = affine_transform(img1_rect, tri1_rect, tri_avg_rect, size)
        warped_img2 = affine_transform(img2_rect, tri2_rect, tri_avg_rect, size)

        # Blend the two triangles
        img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

        # Create a mask
        mask = np.zeros((rect_avg[3], rect_avg[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_avg_rect), (1.0, 1.0, 1.0), 16, 0)

        # Blend the triangles into the output image
        img_morphed[rect_avg[1]:rect_avg[1] + rect_avg[3], rect_avg[0]:rect_avg[0] + rect_avg[2]] = \
            img_morphed[rect_avg[1]:rect_avg[1] + rect_avg[3], rect_avg[0]:rect_avg[0] + rect_avg[2]] * (1 - mask) + \
            img_rect * mask

    # Convert to uint8 and save
    img_morphed = np.uint8(img_morphed)
    cv2.imwrite(output_path, img_morphed)

    # Display the output
    show_img = cv2.cvtColor(img_morphed, cv2.COLOR_BGR2RGB)
    plt.imshow(show_img)
    plt.axis('off')
    plt.show()

    return img_morphed
