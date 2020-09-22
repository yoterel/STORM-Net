import numpy as np
from utils import pairwise


def read_template_file(template_path):
    """
    reads a template file in telaviv format ("sensor x y z rx ry rz") or in princeton format ("name x y z")
    multiple sessions in same file are assumed to be delimited by a line "*" (and first session starts with it)
    note: assumes certain order of capturing in telaviv format (since no names are given)
    :param template_path: the path to template file
    :return: positions is a list of np array per session, names is a list of lists of names per session.
             note: if two sensors exists, they are stacked in a nx2x3 array, else nx3 for positions.
    """
    file_handle = open(str(template_path))
    file_contents = file_handle.read()
    contents_split = file_contents.splitlines()
    non_empty_lines = [line for line in contents_split if line]
    delimiters = [i for i, x in enumerate(non_empty_lines) if x == "*"]
    names = [[]]
    if not delimiters:
        cond = len(non_empty_lines[0].split()) <= 4
        sessions = [non_empty_lines]
    else:
        cond = len(non_empty_lines[delimiters[0]+1].split()) <= 4
        sessions = [non_empty_lines[delimiters[0]+1:delimiters[1]],
                    non_empty_lines[delimiters[1]+1:delimiters[2]],
                    non_empty_lines[delimiters[2]+1:]]
        names = [[], [], []]
    if cond:
        file_format = "princeton"
        if "***" in non_empty_lines[0]:
            non_empty_lines.pop(0)
    else:
        file_format = "telaviv"
    data = []
    if file_format == "telaviv":
        # labeled_names = ['rightear', 'nosebridge', 'nosetip', 'righteye', 'lefteye', 'leftear', 'cz', 'fp1', 'fp2', 'fpz']
        labeled_names = ['leftear', 'nosebridge', 'nosetip', 'lefteye', 'righteye', 'rightear',
                         'f8', 'fp2', 'fpz', 'fp1', 'f7', 'cz', 'o1', 'oz', 'o2']
        # labeled_names = [item for item in labeled_names for i in range(2)]
        for j, session in enumerate(sessions):
            sensor1_data = []
            sensor2_data = []
            for i, (sens1, sens2) in enumerate(pairwise(session)):
                if i < len(labeled_names):
                    name = labeled_names[i]
                else:
                    name = i-len(labeled_names)
                data1 = sens1.split()
                if data1[1] == "?":
                    continue
                names[j].append(name)
                x, y, z = float(data1[1]), float(data1[2]), float(data1[3])
                sensor1_data.append(np.array([x, y, z]))
                data2 = sens2.split()
                x, y, z = float(data2[1]), float(data2[2]), float(data2[3])
                sensor2_data.append(np.array([x, y, z]))
            data.append(np.stack((sensor1_data, sensor2_data), axis=1))
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
            names[0].append(name)
        if 0 not in names[0]:
            end = names[0][-1]
            names[0][names.index(1):] = [x for x in range(end)]
        data = [np.array(data)]
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