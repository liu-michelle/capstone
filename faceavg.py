import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay
import sys
from matplotlib import pyplot as plt

img1 = cv2.imread("./pics/man1.jpg")
img2 = cv2.imread("./pics/man2.jpg")
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
predictor_model = 'shape_predictor_68_face_landmarks.dat'

def get_points(image):
    
    # Use dlib to get the face landmarks(68 points)
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print('No face detected in image {}'.format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])

    # Add 8 image frame coordinate points
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

points1 = get_points(img1)
points2 = get_points(img2)

alpha = 0.5
# Calculate the average coordinates of points in two images
points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)

#define an all-zero matrix to store the merge image
img1 = np.float32(img1)
img2 = np.float32(img2)
img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

def get_triangles(points):
    return Delaunay(points).simplices

#triangles1 = get_triangles(points1)
#triangles2 = get_triangles(points2)
triangles = get_triangles(points)

#affine transform for the Delaunay triangulation result
def affine_transform(input_image, input_triangle, output_triangle, size):
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle))
    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image

for i in triangles:
    
    # Calculate the frame of triangles
    x = i[0]
    y = i[1]
    z = i[2]

    tri1 = [points1[x], points1[y], points1[z]]
    tri2 = [points2[x], points2[y], points2[z]]
    tri = [points[x], points[y], points[z]]
    
    rect1 = cv2.boundingRect(np.float32([tri1]))
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []
    
    for i in range(0, 3):
        tri_rect_warped.append(((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))
    
    # Accomplish the affine transform in triangles
    img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
    warped_img1 = affine_transform(img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(img2_rect, tri_rect2, tri_rect_warped, size)
    
    # Calculate the result based on alpha
    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

    # Generate the mask
    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

    # Accomplish the mask in the merged image
    img_morphed[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img_morphed[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
            rect[2]] * (1 - mask) + img_rect * mask
    
img_morphed = np.uint8(img_morphed)

show_img = cv2.cvtColor(img_morphed, cv2.COLOR_BGR2RGB) 
plt.imshow(show_img)
plt.show()