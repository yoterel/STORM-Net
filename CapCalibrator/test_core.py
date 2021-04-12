import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import geometry
import video
from pathlib import Path
import file_io
import MNI


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
    assert 0 in names
    unsorted_origin_xyz = data[:names.index(0), :]  # non numbered optodes are treated as anchors for projection (they were not calibrated)
    unsorted_origin_names = np.array(names[:names.index(0)])
    others_xyz = data[names.index(0):, :]  # numbered optodes were calibrated, and they will be transformed to MNI
    # these names are written in an order the algorithm expects (and MNI template data was written in)
    target_origin_names = np.array(["nosebridge", "inion", "rightear", "leftear",
                                    "fp1", "fp2", "fz", "f3",
                                    "f4", "f7", "f8", "cz",
                                    "c3", "c4", "t3", "t4",
                                    "pz", "p3", "p4", "t5",
                                    "t6", "o1", "o2"])

    # sort our anchors using the order above
    selected_indices, sorting_indices = np.where(target_origin_names[:, None] == unsorted_origin_names[None, :])
    origin_xyz = unsorted_origin_xyz[sorting_indices]
    otherH, otherC, otherHSD, otherCSD = MNI.project(origin_xyz, others_xyz, selected_indices)
    # np.savez('resource/mni_projection_test.npz', name1=otherH, name2=otherC, name3=otherHSD, name4=otherCSD)
    data = np.load('resource/mni_projection_test.npz')
    otherH_loaded, otherC_loaded, otherHSD_loaded, otherCSD_loaded = data["name1"], data["name2"], data["name3"], data["name4"]
    assert np.all(np.isclose(otherH_loaded, otherH))
    assert np.all(np.isclose(otherC_loaded, otherC))
    assert np.all(np.isclose(otherHSD_loaded, otherHSD))
    assert np.all(np.isclose(otherCSD_loaded, otherCSD))


# if __name__ == "__main__":
#     test_MNI_projection()