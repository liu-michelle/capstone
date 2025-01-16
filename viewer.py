import pyglet
pyglet.options['shadow_window'] = False
import numpy as np
import trimesh
from pyrender import PerspectiveCamera, SpotLight, Mesh, Node, Scene, Viewer, OffscreenRenderer

#==============================================================================
# Mesh creation
#==============================================================================

# Load the fuze 3D model (make sure the path to the .obj file is correct)
fuze_trimesh = trimesh.load('./fuze.obj')
fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

#==============================================================================
# Light creation
#==============================================================================

# Create a spotlight for the scene
spot_l = SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)

#==============================================================================
# Camera creation
#==============================================================================

# Create a perspective camera
cam = PerspectiveCamera(yfov=(np.pi / 3.0))

#==============================================================================
# Scene creation
#==============================================================================

# Create the scene and add ambient light
scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

# Add the fuze mesh to the scene with translation to center it
fuze_node = Node(mesh=fuze_mesh, translation=np.array([0.0, 0.0, 0.0]))  # Center the fuze object
scene.add_node(fuze_node)

# Add the spotlight to the scene
scene.add(spot_l, pose=np.eye(4))

#==============================================================================
# Position the camera to always point at the object (fuze_trimesh)
#==============================================================================

# Calculate the bounding box of the object
bbox = fuze_trimesh.bounding_box
center = bbox.centroid  # Center of the object
extents = bbox.extents  # Size of the object in X, Y, Z directions

# Calculate a good distance for the camera to be away from the object
camera_distance = np.linalg.norm(extents) * 2  # This distance ensures the object is fully visible

# Set the camera position
camera_position = np.array([center[0], center[1], center[2] + camera_distance])

# Define the camera pose (position + orientation)
camera_pose = np.array([
    [1.0, 0.0, 0.0, camera_position[0]],  # Positioning camera at the calculated distance
    [0.0, 1.0, 0.0, camera_position[1]],
    [0.0, 0.0, 1.0, camera_position[2]],
    [0.0, 0.0, 0.0, 1.0]
])

# Add the camera to the scene with the calculated pose
scene.add(cam, pose=camera_pose)

#==============================================================================
# Use the Viewer to view the scene
#==============================================================================

v = Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 800))

