import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
import geometry


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
