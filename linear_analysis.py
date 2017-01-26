import csv
from itertools import groupby
from os import listdir
import linear_fit as l_f
import numpy as np



# FILE_PATH = '/Users/ForeverQ/Downloads/mra2/csv-select/'
FILE_PATH = './'

#
# out: <list> file names of string
#
def generate_csv_file_names(file_path = FILE_PATH):
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
def make_entity_index(csv_filename):
    with open(FILE_PATH + csv_filename) as f:
        f_csv = csv.reader(f)
        data_lists = list(f_csv)
        data_tuples = []
        field_names = data_lists[0]
        for row in data_lists[1:]:

            # get timestamp, entity and value
            data_tuples.append((row[0], row[field_names.index(' entity ')], row[-1]))
        data_tuples_entity_classify = sorted(data_tuples, key=lambda item: item[1])
        dict_by_entity = dict()
        for entity, tuples in groupby(data_tuples_entity_classify, lambda item:item[1]):
            dict_by_entity[entity + '.' + csv_filename] = list(tuples)
    return dict_by_entity


# merge two entities and return a list of tuples with two entities sorted by time
# keynames are like entity.type.csv
# first item is two key names
# in: <string>key1, <dict>keys are entity.type.csv and values are lists of tuples
# out: <list>list of tuples and the first item is (key1,key2)
# add a column of key,key means entity.type.csv which stands more then only entity
#
def merge(key1, dict1, key2, dict2):
    dict1[key1] = [item + (key1,) for item in dict1[key1]]
    dict2[key2] = [item + (key2,) for item in dict2[key2]]
    two_entities = dict1[key1] + dict2[key2]
    return [(key1, key2)] + sorted(two_entities, key=lambda item: item[0])


#
# use the list with two entities to make x and y(return a dict with two items)
# in: <list>list with data of two entities, first item is (key1,key2), in each tuple, columns are (time,entity,value,key)
# out: <dict>a dictionary with two items key1:<list>values, key2:<values>
#

def make_data_pairs(two_entity_list):
    key1, key2 = two_entity_list[0]
    key_entity_table = {key1.split('.')[0]:key1, key2.split('.')[0]:key2}
    x_y_dict = {key1:[], key2:[]}
    key_now = two_entity_list[1][3]
    value_now = float(two_entity_list[1][2])
    value_count = 1
    for row in two_entity_list[2:]:
        if key_now != row[3]:
            x_y_dict[key_now].append(value_now/value_count)
            value_now = float(row[2])
            value_count = 1
            key_now = row[3]
        else:
            value_now += float(row[2])
            value_count += 1
    return x_y_dict

def make_total_dict():
    file_names = generate_csv_file_names()
    dict_total = dict()
    for file_name in file_names:
        dict_now = make_entity_index(file_name)
        dict_total = dict(dict_total, **dict_now)
    return dict_total

if __name__ == '__main__':

    dict_total = make_total_dict()

    # key_names in order by different types
    key_names = sorted(dict_total, key=lambda item: item[::-1])



    # one dimension part
    correlation_count = 0
    for i in range(0,len(key_names)-1):
        for j in range(i+1,len(key_names)):
            first_name = key_names[i]
            second_name = key_names[j]

            # first_name = ' RBR Pressure.Depth.csv'
            # second_name = ' RBR Pressure.Pressure.csv'

            two_entity_list = merge(first_name, dict_total, second_name, dict_total)
            x_y_dict = make_data_pairs(two_entity_list)
            x_values = x_y_dict[first_name]
            y_values = x_y_dict[second_name]

            #
            # except for the all 0 data like ( Altimeter 1)
            if (np.sum(x_values) == 0 or np.sum(y_values) == 0):
                continue

            k, b = l_f.my_leastsq(x_values, y_values)
            r_2 = l_f.get_r_squared2(lambda x:k*x+b,x_values,y_values)
            if r_2 >= 0.85:
                correlation_count += 1
                print(k, b, r_2)
                l_f.my_visualize(first_name, x_values, second_name, y_values, k, b, correlation_count)

    print(correlation_count)