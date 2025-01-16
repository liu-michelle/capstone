import cv2
import mediapipe as mp
import numpy as np
import trimesh

def capture_image():
    """
    Captures an image using the webcam.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 's' to take a picture or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        cv2.imshow("Capture Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to capture
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):  # Press 'q' to quit
            cap.release()
            cv2.destroyAllWindows()
            return None


def generate_3d_mesh(image):
    """
    Generate a 3D mesh with texture mapping from the input image using Mediapipe Face Mesh.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Convert image to RGB as Mediapipe works with RGB images
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the frame and detect faces
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("No faces detected.")
        return None, None, None

    # Get the first detected face landmarks
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape

    # Convert landmarks to 3D coordinates, flipping the y-axis
    vertices = np.array([
        [lm.x * w, h - lm.y * h, lm.z * w]  # Flip y-coordinate
        for lm in face_landmarks.landmark
    ])

    # Extract colors from the original image for each vertex
    colors = np.array([
        rgb_image[int(lm.y * h), int(lm.x * w)] / 255.0 for lm in face_landmarks.landmark
    ])  # Normalize to [0, 1] for Trimesh

    # Convert edges into triangles (simplified based on the Face Mesh Tesselation)
    connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
    triangles = []
    for edge1, edge2 in connections:
        for edge3 in connections:
            if edge3[0] == edge2:  # Find a common vertex to form a triangle
                triangles.append([edge1, edge2, edge3[1]])

    triangles = np.array(triangles)

    return vertices, triangles, colors


def save_obj(vertices, faces, filename="output.obj"):
    """
    Save the 3D vertices and faces as an OBJ file.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)


if __name__ == "__main__":
    print("Starting 3D Face Reconstruction...")

    # Step 1: Capture an image
    image = capture_image()
    if image is None:
        print("No image captured. Exiting.")
        exit()

    # Step 2: Generate 3D mesh
    vertices, triangles, colors = generate_3d_mesh(image)
    if vertices is None or triangles is None or colors is None:
        print("Failed to generate 3D mesh. Exiting.")
        exit()

    # Step 3: Save the 3D mesh as an OBJ file
    save_obj(vertices, triangles)
    print("3D Mesh saved as 'output.obj'.")
