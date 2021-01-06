import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import geometry
import video
from pathlib import Path
import file_io


def test_3d_rigid_transform():
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
    assert (pytest.approx(rmse) == 0)


def test_frame_selection():
    frames, indices = video.video_to_frames(Path("../example_videos/example_video.mp4"),
                                            force_reselect=True,
                                            dump_frames=False)
    assert (frames and indices)


def test_force_frame_selection():
    forced_indices = [x for x in range(10)]
    frames, indices = video.video_to_frames(Path("../example_videos/example_video.mp4"),
                                            frame_indices=forced_indices,
                                            force_reselect=True,
                                            dump_frames=False)
    assert (forced_indices == indices)


def test_MNI_projection():
    names, data, _, _ = file_io.read_template_file(Path("../example_models/example_model.txt"))
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    projected_data = geometry.project_sensors_to_MNI([names, data])
    assert (projected_data is not None)
