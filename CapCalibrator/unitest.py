import unittest
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import geometry

def last_rt(P, Q):
    affine_matrix = cv2.estimateAffine3D(P, Q, confidence=0.99)
    SR = affine_matrix[1][:, :-1]
    T = affine_matrix[1][:, -1]
    s = np.linalg.norm(SR, axis=1)
    S = np.identity(3) * s
    S_divide = np.copy(S)
    S_divide[S_divide == 0] = 1
    R = SR / S_divide
    return T, S, R, SR


class MyTestCase(unittest.TestCase):
    def test_something(self):
        a1 = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0.5, 0.5],
        ])
        r = R.from_euler('z', 90, degrees=True)
        r_mat = r.as_matrix()
        t = [1, 0, 0]
        a2 = (r_mat @ a1.T).T + t
        r_pred, t_pred = geometry.rigid_transform_3d_nparray(a1, a2)
        a2_est = (r_pred @ a1.T).T + t_pred
        rmse = geometry.get_rmse(a2, a2_est)
        self.assertAlmostEqual(rmse, 0)


if __name__ == '__main__':
    unittest.main()
