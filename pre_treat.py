import csv
from itertools import groupby
from os import listdir


# FILE_PATH = '/Users/ForeverQ/Downloads/mra2/csv-select/'
FILE_PATH_S = '/Users/ForeverQ/Programming/svn/hengqian/src/datas/data_1/'
FILE_PATH_M = '/Users/ForeverQ/Programming/svn/hengqian/src/datas/data_1/mult_dm/'


#
# turn the value to float
#
def func_float(item):
    item[0] = float(item[0])
    item[1] = float(item[1])
    return item


#
# out: <list> file names of string
#
def generate_csv_file_names(file_path):
    filename_list = listdir(file_path)
    csv_file_names = []
    for filename in filename_list:
        if filename.endswith('.csv'):
            csv_file_names.append(filename)
    return csv_file_names


#
# process a csv file,distinguish them by entities and return a dictionary and keys are entities.type.csv
# in: <string> a filename of string
# out: <dictionary> key: entities.type.csv, value: <list> a list of tuples of the same type
#
def make_entity_index_S(csv_filename):
    with open(FILE_PATH_S + csv_filename) as f:
        f_csv = csv.reader(f)
        data_lists = list(f_csv)
        data_lists_tev = []
        field_names = data_lists[0]
        for row in data_lists[1:]:

            # get timestamp, entity and value
            data_lists_tev.append([row[0], row[field_names.index(' entity ')], row[-1]])
        data_tuples_entity_classify = sorted(data_lists_tev, key=lambda item: item[1])
        dict_by_entity = dict()
        for entity, items in groupby(data_tuples_entity_classify, lambda item:item[1]):
            whole_list = list(items)
            t_v_list = [[item[0], item[2]] for item in whole_list]
            dict_by_entity[entity + '.' + csv_filename] = t_v_list
    return dict_by_entity


def make_entity_index_M(csv_filename):
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
    file_names_S = generate_csv_file_names(file_path=FILE_PATH_S)
    file_names_M = generate_csv_file_names(file_path=FILE_PATH_M)
    dict_total = dict()
    for file_name in file_names_S:
        dict_now = make_entity_index_S(file_name)
        dict_total = dict(dict_total, **dict_now)
    for file_name in file_names_M:
        dict_now = make_entity_index_M(file_name)
        dict_total = dict(dict_total, **dict_now)

    # turn the data from str to float
    for key_name in dict_total:
         dict_total[key_name] = [*map(func_float, dict_total[key_name])]
    return dict_total


# merge two entities and return a list of tuples with two entities sorted by time
# keynames are like entity.type.csv
# first item is two key names
# in: <string>key1, <dict>keys are entity.type.csv and values are lists of tuples
# out: <list>list of tuples and the first item is (key1,key2)
# add a column of key,key means entity.type.csv which stands more then only entity
#
def merge(key1, key2, dict):
    two_entities = [item + [key1] for item in dict[key1]] + [item + [key2] for item in dict[key2]]
    return [[key1, key2]] + sorted(two_entities, key=lambda item: item[0])


#
# use the list with two entities to make x and y(return a dict with two items)
# in: <list>list with data of two entities, first item is (key1,key2), in each tuple, columns are (time,entity,value,key)
# out: <dict>a dictionary with two items key1:<list>values, key2:<values>
#
def make_data_pairs(two_entity_list):
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
