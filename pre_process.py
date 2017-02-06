import csv
from itertools import groupby
from os import listdir


# These are the paths of data.
# FILE_PATH_S stores the one-dimension data
# FILE_PATH_M stores the mult-dimension data
FILE_PATH_S = '/Users/ForeverQ/Programming/svn/hengqian/src/datas/data_1/'
FILE_PATH_M = '/Users/ForeverQ/Programming/svn/hengqian/src/datas/data_1/mult_dm/'


def func_float(item):
    """Turns string values into floats"""
    item[0] = float(item[0])
    item[1] = float(item[1])
    return item


def generate_csv_file_names(file_path):
    """Generates a list of file names of string under the file path"""
    filename_list = listdir(file_path)
    csv_file_names = []
    for filename in filename_list:
        if filename.endswith('.csv'):
            csv_file_names.append(filename)
    return csv_file_names


def make_entity_index_S(csv_filename):
    """Extracts different entities of values from one csv file of one dimension data

    Processes one csv file and makes different entities as different groups of values

    Arg:
        csv_filename: A string which represents the filename of a csv file

    Returns:
        A dict mapping keys of entities' names to the corresponding list row extracted.
        Each row is represented as a list of timestamp and value. For example:

        {'Distance Chan 1.Distance.csv': [[1471014284.771, 0.775], [1471014285.694, 0.775], ...],
         'Distance Chan 2.Distance.csv': [[1471014284.771, 0.772], [1471014285.694, 0.771], ...],
         'Distance Chan 3.Distance.csv': [[1471014284.771, 0.785], [1471014285.694, 0.784], ...],
         ...}
    """
    with open(FILE_PATH_S + csv_filename) as f:
        f_csv = csv.reader(f)
        data_lists = list(f_csv)
        data_lists_tev = []
        field_names = data_lists[0]
        for row in data_lists[1:]:
            data_lists_tev.append([row[0], row[field_names.index(' entity ')], row[-1]])
        data_tuples_entity_classify = sorted(data_lists_tev, key=lambda item: item[1])
        dict_by_entity = dict()
        for entity, items in groupby(data_tuples_entity_classify, lambda item:item[1]):
            whole_list = list(items)
            t_v_list = [[item[0], item[2]] for item in whole_list]
            dict_by_entity[entity + '.' + csv_filename] = t_v_list
    return dict_by_entity


def make_entity_index_M(csv_filename):
    """Extracts values from one csv file of multi-dimension data

    Processes one csv file and makes each dimension data as different groups of values.
    Since these multi-dimension data are all from the same entity of "Xsens", it's enough
    to distinguish them by key of "dimension-name.filename.csv"

    Arg:
        csv_filename: A string which represents the filename of a csv file

    Returns:
        A dict mapping keys of dimensions' names to the corresponding list row extracted.
        Each row is represented as a list of timestamp and value. For example:

        {'x (m/s/s).Acceleration.csv': [[1471014260.316, -0.04102239], [1471014260.336, -0.05046942], ...],
         'y (m/s/s).Acceleration.csv': [[1471014260.316, 0.00478985], [1471014260.336, 0.00274967], ...],
         'z (m/s/s).Acceleration.csv': [[1471014260.316, 0.00446284], [1471014260.336, 0.00052702], ...],
         ...}
    """
    with open(FILE_PATH_M + csv_filename) as f:
        f_csv = csv.reader(f)
        data_lists = list(f_csv)
        field_names = data_lists[0]
        dict_by_entity = dict()
        if csv_filename == 'EulerAngles.csv':
            for i in range(4):
                data_lists_tv = []
                for row in data_lists[1:]:
                    data_lists_tv.append([row[0], row[-1-i]])
                dict_by_entity[field_names[-1-i] + '.' + csv_filename] = data_lists_tv
        else:
            for i in range(3):
                data_lists_tv = []
                for row in data_lists[1:]:
                    data_lists_tv.append([row[0], row[-1-i]])
                dict_by_entity[field_names[-1-i] + '.' + csv_filename] = data_lists_tv
    return dict_by_entity


def make_total_dict():
    """Integrate data of both one-dimension and multi-dimension to one dict"""
    file_names_S = generate_csv_file_names(file_path=FILE_PATH_S)
    file_names_M = generate_csv_file_names(file_path=FILE_PATH_M)
    dict_total = dict()
    for file_name in file_names_S:
        dict_now = make_entity_index_S(file_name)
        dict_total = dict(dict_total, **dict_now)
    for file_name in file_names_M:
        dict_now = make_entity_index_M(file_name)
        dict_total = dict(dict_total, **dict_now)

    # Turns the data value from str to float
    for key_name in dict_total:
         dict_total[key_name] = [*map(func_float, dict_total[key_name])]
    return dict_total


def merge(key1, key2, dict):
    """Merges values of two entities into one"""
    two_entities = [item + [key1] for item in dict[key1]] + [item + [key2] for item in dict[key2]]
    return [[key1, key2]] + sorted(two_entities, key=lambda item: item[0])


def make_data_pairs(two_entity_list):
    """make value pairs of two entities by time

    Arg:
        two_entity_list: Data from two entities in one list

    Returns:
        A dict with three elements which have the same length. It's like

        {'time':       [value_1, value_2, ..., value_n],
         'key_name_1': [value_1, value_2, ..., value_n],
         'key_name_2': [value_1, value_2, ..., value_n]}
    """
    key1, key2 = two_entity_list[0]
    x_y_dict = {'time':[], key1:[], key2:[]}
    number_of_key1 = len(list(filter(lambda item:item[2]==key1, two_entity_list[1:])))
    number_of_key2 = len(list(filter(lambda item:item[2]==key2, two_entity_list[1:])))
    if number_of_key1 <= number_of_key2:
        key_based = key1
        key_follow = key2
    else:
        key_based = key2
        key_follow = key1
    for i in range(2, len(two_entity_list) - 1):
        if two_entity_list[i][2] == key_based:
            if two_entity_list[i-1][2] == key_based and two_entity_list[i+1][2] != key_based:
                x_y_dict['time'].append(two_entity_list[i][0])
                x_y_dict[key_based].append(two_entity_list[i][1])
                x_y_dict[key_follow].append(two_entity_list[i+1][1])
            elif two_entity_list[i-1][2] != key_based and two_entity_list[i+1][2] == key_based:
                x_y_dict['time'].append(two_entity_list[i][0])
                x_y_dict[key_based].append(two_entity_list[i][1])
                x_y_dict[key_follow].append(two_entity_list[i-1][1])
            elif two_entity_list[i-1][2] != key_based and two_entity_list[i+1][2] != key_based:
                if (abs(two_entity_list[i-1][0]-two_entity_list[i][0]) <=
                    abs(two_entity_list[i+1][0]-two_entity_list[i][0])):
                    x_y_dict['time'].append(two_entity_list[i][0])
                    x_y_dict[key_based].append(two_entity_list[i][1])
                    x_y_dict[key_follow].append(two_entity_list[i-1][1])
                else:
                    x_y_dict['time'].append(two_entity_list[i][0])
                    x_y_dict[key_based].append(two_entity_list[i][1])
                    x_y_dict[key_follow].append(two_entity_list[i+1][1])
    return x_y_dict
