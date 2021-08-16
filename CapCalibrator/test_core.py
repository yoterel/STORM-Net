import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import geometry
import video
from pathlib import Path
import file_io
import MNI
import torch_src.MNI_torch as MNI_torch
import render
import torch

@pytest.fixture
def anchors_and_sensors():
    names, data, _, _ = file_io.read_template_file(Path("../example_models/example_model.txt"))
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    assert 0 in names
    unsorted_origin_xyz = data[:names.index(0),
                          :]  # non numbered optodes are treated as anchors for projection (they were not calibrated)
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
    return [origin_xyz, others_xyz, selected_indices]

@pytest.fixture
def network():
    import torch_src.torch_model as torch_model

    class Options():
        def __init__(self):
            self.network_output_size = 3
            self.template = Path("../example_models/example_model.txt")
            self.architecture = "fc"
            self.loss = "l2"
            self.device = "cpu"
    opt = Options()
    network = torch_model.MyNetwork(opt)
    return network

@pytest.fixture
def dataloader():
    import torch_src.torch_data as torch_data

    class Options():
        def __init__(self):
            self.data_path = Path("/disk1/yotam/capnet/scene3_100k")
            self.is_train = True
            self.template = Path("../example_models/example_model.txt")
            self.device = "cuda:7"
            self.loss = "l2+projection"

    opt = Options()
    loader = torch_data.MyDataSet(opt)
    return loader


def test_raw_data_set(dataloader, anchors_and_sensors):
    origin_xyz, others_xyz, selected_indices = anchors_and_sensors
    device = "cuda:7"
    x, y = dataloader.__getitem__(0)
    y_euler = y["rot_and_scale"]
    y_pc = y["raw_projected_data"]

    y_euler_numpy = y_euler.cpu().numpy()
    rot = R.from_euler('xyz', list(y_euler_numpy), degrees=True)
    rot_mat = rot.as_matrix()
    transformed_others_xyz = (rot_mat @ others_xyz.T).T

    origin_xyz_torch = torch.from_numpy(origin_xyz).float().to(device)
    transformed_others_xyz_torch = torch.from_numpy(transformed_others_xyz).float().to(device)
    selected_indices_torch = torch.from_numpy(selected_indices).to(device)
    torch_mni, _, _ = MNI_torch.torch_project_non_differentiable(origin_xyz_torch,
                                                                 transformed_others_xyz_torch.unsqueeze(0),
                                                                 selected_indices_torch)
    assert torch.all(torch.isclose(y_pc, torch_mni))




def test_mni_ours_vs_naive_vs_full(anchors_and_sensors):
    origin_xyz, others_xyz, selected_indices = anchors_and_sensors
    device = "cuda:7"
    euler = (np.random.rand(3) * 10) - 5
    rot = R.from_euler('xyz', list(euler), degrees=True)
    rot_mat = rot.as_matrix()
    transformed_others_xyz = (rot_mat @ others_xyz.T).T
    origin_xyz_torch = torch.from_numpy(origin_xyz).float().to(device)
    transformed_others_xyz_torch = torch.from_numpy(transformed_others_xyz).float().to(device)
    selected_indices_torch = torch.from_numpy(selected_indices).to(device)
    torch_mni, errors, naive_torch_mni = MNI_torch.torch_project_non_differentiable(origin_xyz_torch,
                                                                               transformed_others_xyz_torch.unsqueeze(0),
                                                                               selected_indices_torch, output_errors=True)
    diff_torch_mni_projection = MNI_torch.torch_project(origin_xyz_torch,
                                                        transformed_others_xyz_torch.unsqueeze(0),
                                                        selected_indices_torch)

    naive_to_full = torch.mean(torch.linalg.norm(naive_torch_mni - torch_mni, dim=1))
    ours_to_full = torch.mean(torch.linalg.norm(diff_torch_mni_projection.squeeze(0) - torch_mni, dim=1))
    ours_to_naive = torch.mean(torch.linalg.norm(diff_torch_mni_projection.squeeze(0) - naive_torch_mni, dim=1))
    assert True


def test_differentiable_find_affine(anchors_and_sensors):
    """
    tests how close the differentiable version of finding the affine transformations is to the original
    :param anchors_and_sensors:
    :return:
    """
    origin_xyz, others_xyz, selected_indices = anchors_and_sensors
    refN = 17  # number of reference brains
    pointN = others_xyz.shape[0]  # number of sensors to project
    classic_result = MNI.find_affine_transforms(origin_xyz, others_xyz, selected_indices, refN, pointN)
    origin_xyz = torch.from_numpy(origin_xyz).float()
    others_xyz = torch.from_numpy(others_xyz).float()
    selected_indices = torch.from_numpy(selected_indices)
    differentiable_result = MNI_torch.torch_find_affine_transforms(origin_xyz, others_xyz, selected_indices, refN,
                                                                   pointN)
    test1 = classic_result.astype(np.float32)
    test2 = differentiable_result.numpy()
    assert np.all(np.isclose(test1, test2, atol=1e-5))


def test_mni_torch_vs_mni_numpy(anchors_and_sensors):
    """
    tests if mni projection using torch is the same as nump
    :param anchors_and_sensors:
    :return:
    """
    origin_xyz, others_xyz, selected_indices = anchors_and_sensors
    devices = ["cpu", "cuda:7"]
    for device in devices:
        euler = (np.random.rand(3) * 10) - 5
        rot = R.from_euler('xyz', list(euler), degrees=True)
        rot_mat = rot.as_matrix()
        transformed_others_xyz = (rot_mat @ others_xyz.T).T
        _, np_mni, _, np_mni_sd = MNI.project(origin_xyz, transformed_others_xyz, selected_indices, output_errors=True)
        origin_xyz_torch = torch.from_numpy(origin_xyz).float().to(device)
        transformed_others_xyz_torch = torch.from_numpy(transformed_others_xyz).float().to(device)
        selected_indices_torch = torch.from_numpy(selected_indices).to(device)
        torch_mni, torch_mni_sd, _ = MNI_torch.torch_project_non_differentiable(origin_xyz_torch,
                                                                                transformed_others_xyz_torch.unsqueeze(0),
                                                                                selected_indices_torch,
                                                                                output_errors=True)
        assert torch.all(torch.isclose(torch.from_numpy(np_mni).float().to(device), torch_mni.squeeze(0)))
        assert torch.all(torch.isclose(torch.from_numpy(np_mni_sd).float().to(device), torch_mni_sd.squeeze(0), atol=1e-4))



def test_differentiable_mni_optimization(network, anchors_and_sensors):
    """
    tests if optimizing through torch yields good gradients
    :param network:
    :param anchors_and_sensors:
    :return:
    """
    Niter = 1
    origin_xyz, others_xyz, selected_indices = anchors_and_sensors
    euler = (np.random.rand(3) * 10) - 5
    rot = R.from_euler('xyz', list(euler), degrees=True)
    rot_mat = rot.as_matrix()
    transformed_others_xyz = (rot_mat @ others_xyz.T).T
    _, gt, _, _ = MNI.project(origin_xyz, transformed_others_xyz, selected_indices)
    euler_torch = torch.full([1, 3], 1.0, device="cpu", requires_grad=True)
    init_diff = np.linalg.norm(euler - euler_torch.cpu().detach().numpy())
    optimizer = torch.optim.SGD([euler_torch], lr=1.0, momentum=0.9)
    for i in range(Niter):
        # Initialize optimizer
        optimizer.zero_grad()

        origin_xyz_torch = torch.from_numpy(origin_xyz).float()
        others_xyz_torch = torch.from_numpy(others_xyz).float()
        torch_matrcies = network.euler_to_matrix(euler_torch)
        transformed_sensors = torch.transpose(torch.bmm(torch_matrcies, others_xyz_torch.T.repeat(1, 1, 1)), 1, 2)
        projected_out = MNI_torch.torch_project(origin_xyz_torch, transformed_sensors, selected_indices,
                                                resource_folder="resource")
        loss = torch.mean(torch.linalg.norm(torch.from_numpy(gt).unsqueeze(0) - projected_out, dim=2))
        # print(loss.cpu().detach().numpy())
        # print("euler:", euler)
        # print("euler_torch:", euler_torch.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
    final_diff = np.linalg.norm(euler - euler_torch.cpu().detach().numpy())
    assert final_diff < init_diff


def test_batched_euler_to_matrix(network):
    """
    tests conversion form euler angles to matrix representation
    :param network:
    :return:
    """
    batch_size = 16
    euler = (np.random.rand(batch_size, 3) * 10) - 5
    scipy_matrices = np.empty((batch_size, 3, 3))
    for i in range(batch_size):
        rot = R.from_euler('xyz', list(euler[i]), degrees=True)
        rot_mat = rot.as_matrix()
        scipy_matrices[i] = rot_mat
    torch_euler = torch.from_numpy(euler)
    torch_matrcies = network.euler_to_matrix(torch_euler)
    assert torch.all(torch.isclose(torch_matrcies, torch.from_numpy(scipy_matrices).float()))


def test_differentiable_MNI_projection(anchors_and_sensors):
    """
    tests how close the differentiable version of mni projection is to the actual algorithm
    :param anchors_and_sensors:
    :return:
    """
    origin_xyz, others_xyz, selected_indices = anchors_and_sensors
    selected_indices_torch = torch.from_numpy(selected_indices)
    euler = (np.random.rand(3) * 10) - 5
    rot = R.from_euler('xyz', list(euler), degrees=True)
    rot_mat = rot.as_matrix()
    transformed_others_xyz = (rot_mat @ others_xyz.T).T
    _, test1, _, _ = MNI.project(origin_xyz, transformed_others_xyz, selected_indices)
    origin_xyz_torch = torch.from_numpy(origin_xyz).float()
    transformed_others_xyz_torch = torch.from_numpy(transformed_others_xyz).float()
    test2 = MNI_torch.torch_project(origin_xyz_torch, transformed_others_xyz_torch.unsqueeze(0), selected_indices_torch)
    print("mean:", torch.mean(torch.linalg.norm(torch.from_numpy(test1) - test2.squeeze(0), dim=1)))
    print(torch.linalg.norm(torch.from_numpy(test1) - test2.squeeze(0), dim=1))
    assert torch.mean(torch.linalg.norm(torch.from_numpy(test1) - test2.squeeze(0), dim=1)) < 1.5


def test_3d_rigid_transform():
    """
    tests rigid transform based on svd
    :return:
    """
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
    """
    tests if the numpy implementation yields same results as matlab one
    :return:
    """
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
    otherH, otherC, otherHSD, otherCSD = MNI.project(origin_xyz, others_xyz, selected_indices, output_errors=True)
    # np.savez('resource/mni_projection_test.npz', name1=otherH, name2=otherC, name3=otherHSD, name4=otherCSD)
    data = np.load('resource/mni_projection_test.npz')
    otherH_loaded, otherC_loaded, otherHSD_loaded, otherCSD_loaded = data["name1"], data["name2"], data["name3"], data["name4"]
    assert np.all(np.isclose(otherH_loaded, otherH))
    assert np.all(np.isclose(otherC_loaded, otherC))
    assert np.all(np.isclose(otherHSD_loaded, otherHSD))
    assert np.all(np.isclose(otherCSD_loaded, otherCSD))


def test_render():
    names, data, file_format, _ = file_io.read_template_file(Path("../example_models/example_model.txt"))
    data = data[0]  # select first (and only) session
    if file_format == "telaviv":
        data = data[:, 0, :]  # select first sensor
    names = names[0]  # select first (and only) session
    render_dir = Path("cache/renders")
    render_dir.mkdir(parents=True, exist_ok=True)
    file_io.delete_content_of_folder(render_dir, subfolders_also=False)
    status, process = render.render(names,
                                    data,
                                    render_dir,
                                    Path("../DataSynth/build/DataSynth.exe"),
                                    Path("cache/log"),
                                    1,
                                    False,
                                    False)
    assert status
    exit_code = process.wait()
    X, Y = file_io.load_raw_json_db(render_dir)
    assert X.shape == (1, 10, 14)
    assert Y.shape == (1, 3)
