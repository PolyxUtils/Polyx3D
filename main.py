import pyxel
import numpy as np
from math import radians, sqrt, cos, sin, tan

def WorldToViewMatrice(camera, at):
    NormalizeVector = lambda v : v / np.linalg.norm(v)

    up = np.array([0, 1, 0])

    w = NormalizeVector(camera - at)
    u = NormalizeVector(np.cross(w, up))
    v = np.cross(w, u)

    return np.array([
        [u[0], u[1], u[2], -np.dot(camera, u)],
        [v[0], v[1], v[2], -np.dot(camera, v)],
        [w[0], w[1], w[2], -np.dot(camera, w)],
        [0, 0, 0, 1]
    ])

def ViewToClipMatrice(fov, aspect):
    far = 100
    near = 0.1

    f = 1 / tan(radians(fov) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2*far*near) / (near - far)],
        [0, 0, -1, 0]
    ])

def ClipToScreenMatrice(w, h):
    return np.array([
        [w / 2, 0, 0, (w - 1) / 2],
        [0, h / 2, 0, (h - 1) / 2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

class App:
    def __init__(self):
        self.triangles = np.array([
            # Front face
            [[-1, -1, 1, 1], [1, -1, 1, 1], [1, 3, 1, 1]],
            [[-1, -1, 1, 1], [1, 3, 1, 1], [-1, 3, 1, 1]],
            # Back face
            [[-1, -1, -1, 1], [1, -1, -1, 1], [1, 1, -1, 1]],
            [[-1, -1, -1, 1], [1, 3, -1, 1], [-1, 3, -1, 1]],
            # Left face
            [[-1, -1, -1, 1], [-1, -1, 1, 1], [-1, 3, 1, 1]],
            [[-1, -1, -1, 1], [-1, 3, 1, 1], [-1, 3, -1, 1]],
            # Right face
            [[1, -1, -1, 1], [1, -1, 1, 1], [1, 3, 1, 1]],
            [[1, -1, -1, 1], [1, 3, 1, 1], [1, 3, -1, 1]],
            # Top face
            [[-1, 3, -1, 1], [1, 3, -1, 1], [1, 3, 1, 1]],
            [[-1, 3, -1, 1], [1, 3, 1, 1], [-1, 3, 1, 1]],
            # Bottom face
            [[-1, -1, -1, 1], [1, -1, -1, 1], [1, -1, 1, 1]],
            [[-1, -1, -1, 1], [1, -1, 1, 1], [-1, -1, 1, 1]],

            [[-6, -1, 1, 1], [-4, -1, 1, 1], [-4, 3, 1, 1]],
            [[-6, -1, 1, 1], [-4, 3, 1, 1], [-6, 3, 1, 1]],
            # Back face
            [[-6, -1, -1, 1], [-4, -1, -1, 1], [-4, 1, -1, 1]],
            [[-6, -1, -1, 1], [-4, 3, -1, 1], [-6, 3, -1, 1]],
            # Left face
            [[-6, -1, -1, 1], [-6, -1, 1, 1], [-6, 3, 1, 1]],
            [[-6, -1, -1, 1], [-6, 3, 1, 1], [-6, 3, -1, 1]],
            # Right face
            [[-4, -1, -1, 1], [-4, -1, 1, 1], [-4, 3, 1, 1]],
            [[-4, -1, -1, 1], [-4, 3, 1, 1], [-4, 3, -1, 1]],
            # Top face
            [[-6, 3, -1, 1], [-4, 3, -1, 1], [-4, 3, 1, 1]],
            [[-6, 3, -1, 1], [-4, 3, 1, 1], [-6, 3, 1, 1]],
            # Bottom face
            [[-6, -1, -1, 1], [-4, -1, -1, 1], [-4, -1, 1, 1]],
            [[-6, -1, -1, 1], [-4, -1, 1, 1], [-6, -1, 1, 1]]
        ])
        
        self.color = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ]

        self.camera = np.array([3.0, 1.0, 0.0])
        self.at = np.array([-1.0, 1.0, 0.0])

        self.yaw = 180
        self.pitch = 0

        pyxel.init(256, 256)
        pyxel.run(self.update, self.draw)

    def update(self):
        speed = 0.1

        old_pose = self.camera.copy()

        rad_yaw = radians(self.yaw)
        rad_pitch = radians(self.pitch)

        if pyxel.btn(pyxel.KEY_Z):  # Move forward
            self.camera[0] += cos(rad_yaw) * speed
            self.camera[2] += sin(rad_yaw) * speed
        if pyxel.btn(pyxel.KEY_S):  # Move backward
            self.camera[0] -= cos(rad_yaw) * speed
            self.camera[2] -= sin(rad_yaw) * speed
        if pyxel.btn(pyxel.KEY_Q):  # Strafe left
            self.camera[0] -= sin(rad_yaw) * speed
            self.camera[2] += cos(rad_yaw) * speed
        if pyxel.btn(pyxel.KEY_D):  # Strafe right
            self.camera[0] += sin(rad_yaw) * speed
            self.camera[2] -= cos(rad_yaw) * speed
            
            
        if pyxel.btn(pyxel.KEY_SPACE):
            self.camera[1] += 0.2
        
        self.shift = pyxel.btn(pyxel.KEY_SHIFT)

        if pyxel.btn(pyxel.KEY_M):
            self.yaw -= 1
        if pyxel.btn(pyxel.KEY_J):
            self.yaw += 1
        if pyxel.btn(pyxel.KEY_I):
            self.pitch += 1
        if pyxel.btn(pyxel.KEY_K):
            self.pitch -= 1

        if self.camera[1] >= 1.05:
            self.camera[1] -= 0.05
        else:
            self.camera[1] = 1

        self.at = np.array([cos(rad_yaw) * cos(rad_pitch), sin(rad_pitch), sin(rad_yaw) * cos(rad_pitch)]) + self.camera

    def draw(self):
        pyxel.cls(0)
        
        if self.shift:
            self.camera[1] -= 0.5
            self.at[1] -= 0.5
        
        WorldToView = WorldToViewMatrice(self.camera, self.at)
        ViewToClip = ViewToClipMatrice(60, 1)
        ViewToScreen = ClipToScreenMatrice(256, 256)
        TransformMatrix = ViewToClip @ WorldToView

        trianglesToShow = []
        for c, t in zip(self.color, self.triangles):
            p1, p2, p3 = TransformMatrix @ t[0], TransformMatrix @ t[1], TransformMatrix @ t[2]
            p1, p2, p3 = p1 / p1[3], p2 / p2[3], p3 / p3[3]
            
            if not ((1 >= p1[0] >= -1 and
                    1 >= p1[1] >= -1 and
                    1 >= p1[2] >= -1) or
                    (1 >= p2[0] >= -1 and
                    1 >= p2[1] >= -1 and
                    1 >= p2[2] >= -1) or
                    (1 >= p3[0] >= -1 and
                    1 >= p3[1] >= -1 and
                    1 >= p3[2] >= -1 )):
                continue

            p1, p2, p3 = ViewToScreen @ p1, ViewToScreen @ p2, ViewToScreen @ p3

            trianglesToShow.append((p1, p2, p3, c))

        trianglesToShow.sort(key=lambda pl : max(pl[0][2], pl[1][2], pl[2][2]), reverse=True)

        for p1, p2, p3, c in trianglesToShow:
            pyxel.tri(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], c)

App()
