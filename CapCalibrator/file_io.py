import numpy as np
from utils import pairwise


def read_template_file(template_path):
    """
    reads a template file in telaviv format ("sensor x y z rx ry rz") or in princeton format ("name x y z")
    note: assumes certain order of capturing in telaviv format (since no names are given)
    :param template_path: the path to template file
    :return: names & position of points in data (if two sensors exists, they are stacked in a nx2x3 array, else nx3)
    """
    file_handle = open(str(template_path))
    file_contents = file_handle.read()
    contents_split = file_contents.splitlines()
    non_empty_lines = [line for line in contents_split if line]
    if len(non_empty_lines[0].split()) <= 4:
        file_format = "princeton"
        if "***" in non_empty_lines[0]:
            non_empty_lines.pop(0)
    else:
        file_format = "telaviv"
    names = []
    data = []
    if file_format == "telaviv":
        sensor1_data = []
        sensor2_data = []
        labeled_names = ['rightear', 'nosebridge', 'nosetip', 'righteye', 'lefteye', 'leftear', 'cz', 'fp1', 'fp2', 'fpz']
        # labeled_names = [item for item in labeled_names for i in range(2)]
        for i, (sens1, sens2) in enumerate(pairwise(non_empty_lines)):
            if i < len(labeled_names):
                name = labeled_names[i]
            else:
                name = i-len(labeled_names)
            names.append(name)
            data1 = sens1.split()
            x, y, z = float(data1[1]), float(data1[2]), float(data1[3])
            sensor1_data.append(np.array([x, y, z]))
            data2 = sens2.split()
            x, y, z = float(data2[1]), float(data2[2]), float(data2[3])
            sensor2_data.append(np.array([x, y, z]))
        data = np.stack((sensor1_data, sensor2_data), axis=1)
    else:  # princeton
        for line in contents_split:
            name, x, y, z = line.split()
            x = float(x)
            y = float(y)
            z = float(z)
            data.append(np.array([x, y, z]))
            try:
                name = int(name)
            except ValueError as verr:
                name = name.lower()
            names.append(name)
        if 0 not in names:
            end = names[-1]
            names[names.index(1):] = [x for x in range(end)]
        data = np.array(data)
    return names, data, file_format


def save_results(data, output_file, v):
    """
    saves data into output file
    :param data:
    :param output_file:
    :param v:
    :return:
    """
    if v:
        print("Saving result to output file.")
    if not output_file:
        output_file = "output.txt"
    # np.savetxt(output_file, data, delimiter=" ")