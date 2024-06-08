import cv2
import dlib
import numpy as np

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("static/img/shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()])

def affine_transform(image1, from_points, to_points, size):
    matrix = cv2.getAffineTransform(np.float32(from_points), np.float32(to_points))
    return cv2.warpAffine(image1, matrix, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def apply_delaunay_triangulation(image1, image2, points1, points2):
    rect = (0, 0, image2.shape[1], image2.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv, ensuring they are within image bounds
    for p in points2:
        if 0 <= p[0] < image2.shape[1] and 0 <= p[1] < image2.shape[0]:
            subdiv.insert((float(p[0]), float(p[1])))
        else:
            print(f"Point {p} is out of bounds for image2 with shape {image2.shape}")

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    # Filter out triangles that have points outside the image bounds
    valid_triangles = []
    for t in triangles:
        pts = [(t[i * 2], t[i * 2 + 1]) for i in range(3)]
        if all(0 <= pt[0] < image2.shape[1] and 0 <= pt[1] < image2.shape[0] for pt in pts):
            valid_triangles.append(t)
    triangles = np.array(valid_triangles, dtype=np.int32)

    for t in triangles:
        pts1 = []
        pts2 = []
        for i in range(0, 3):
            pts1.append(points1[np.where((points2 == (t[i * 2], t[i * 2 + 1])).all(axis=1))[0][0]])
            pts2.append((t[i * 2], t[i * 2 + 1]))

        warp_image = affine_transform(image1, pts1, pts2, (image2.shape[1], image2.shape[0]))
        mask = np.zeros(image2.shape, dtype=image2.dtype)
        cv2.fillConvexPoly(mask, np.int32(pts2), (1, 1, 1), 16, 0)
        image2 = image2 * (1 - mask) + warp_image * mask

    return image2
