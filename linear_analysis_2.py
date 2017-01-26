import csv
from itertools import groupby
from os import listdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np



# FILE_PATH = '/Users/ForeverQ/Downloads/mra2/csv-select/'
FILE_PATH_S = './datas/'
FILE_PATH_M = './datas/mult_dm/'

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


def func_float(item):
    item[0] = float(item[0])
    item[1] = float(item[1])
    return item

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

    two_entity_list = [two_entity_list[0]] + [*map(func_float, two_entity_list[1:])]


    for i in range(2, len(two_entity_list) - 1):
        if two_entity_list[i][2] == key_based:
            if (two_entity_list[i-1][2] == key_based) and (two_entity_list[i+1][2] != key_based):
                x_y_dict['time'].append(two_entity_list[i][0])
                x_y_dict[key_based].append(two_entity_list[i][1])
                x_y_dict[key_follow].append(two_entity_list[i+1][1])
            elif (two_entity_list[i-1][2] != key_based) and (two_entity_list[i+1][2] == key_based):
                x_y_dict['time'].append(two_entity_list[i][0])
                x_y_dict[key_based].append(two_entity_list[i][1])
                x_y_dict[key_follow].append(two_entity_list[i-1][1])
            elif (two_entity_list[i-1][2] != key_based) and (two_entity_list[i+1][2] != key_based):
                if (abs(two_entity_list[i-1][0]-two_entity_list[i][0]) <= abs(two_entity_list[i+1][0]-two_entity_list[i][0])):
                    x_y_dict['time'].append(two_entity_list[i][0])
                    x_y_dict[key_based].append(two_entity_list[i][1])
                    x_y_dict[key_follow].append(two_entity_list[i-1][1])
                else:
                    x_y_dict['time'].append(two_entity_list[i][0])
                    x_y_dict[key_based].append(two_entity_list[i][1])
                    x_y_dict[key_follow].append(two_entity_list[i+1][1])
    return x_y_dict


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
    return dict_total


#
# X and y are pd.DataFrame
# coef is np.array, intercept is np.float64
#
# Is not using this func now!
def get_r_squared2(coef , intercept, X, y):


    X_np = X.as_matrix()
    y_np = y.as_matrix()

    y_bar = np.sum(y_np) / len(y_np)
    ss_t = np.sum(list(map(lambda a,b: (a-b)**2, y_np,[y_bar]*len(y_np))))
    Y_np = np.array(list(map(lambda a: np.dot(coef.T, a) + intercept, X_np)))
    ss_r = np.sum(list(map(lambda a,b: (a-b)**2, Y_np, [y_bar]*len(y_np))))
    return ss_r / ss_t


def make_linear_fitting_pairs(dict_total, key_names):
    count = 0
    # with open('data_pairs_parameters_2.txt','w') as f:
    # with open('data_pairs_parameters_2.csv', 'w') as f:
    #     csv_writer = csv.writer(f, dialect='excel')
    #     csv_writer.writerow(['X', 'y', 'r^2', 'A', 'b'])
    for i in range(0,len(key_names)):
        for j in range(0,len(key_names)):
            if key_names[i] == key_names[j]:
                continue
            first_name = key_names[i]
            second_name = key_names[j]

            # first_name = ' Servo 4.Temperature.csv'
            # second_name = ' z (m/s/s).Acceleration.csv'

            two_entity_list = merge(first_name, second_name, dict_total)
            x_y_dict = make_data_pairs(two_entity_list)

            # about pandas dataframe
            pd.set_option('precision',8)
            df = pd.DataFrame(x_y_dict)
            df.sort_values(by='time')
            # print(df.head())

            # start regression
            x_col = [first_name]
            y_col = [second_name]
            X = df[x_col]
            y = df[y_col]
            lin_reg = LinearRegression()
            model = lin_reg.fit(X, y)

            # A and b
            coef = lin_reg.coef_
            intercept = lin_reg.intercept_

            # r-square
            r_2 = model.score(X, y)
            # r_2_ = get_r_squared2(coef, intercept, X, y)

            # if r_2 <= 0.2:
            count += 1
            print('X:' + first_name)
            print('y:' + second_name)
            print('a=' + str(coef), 'b=' + str(intercept))
            print('r_2: ' + str(r_2))
            print()

            sns.pairplot(df, x_vars=[first_name], y_vars=second_name , size=10, aspect=1.2, kind='reg')
            plt.show()
            # plt.savefig(first_name + ' && ' + second_name + '.png', dpi=150)


            # write to a file
            # f.write('X:' + first_name + '\n')
            # f.write('y:' + second_name + '\n')
            # f.write('a=' + str(coef) + '  b=' + str(intercept) + '\n')
            # f.write('r_2: ' + str(r_2) + '\n\n')

            # write to a csv
            # csv_writer.writerow([first_name, second_name, r_2, coef[0][0], intercept[0]])

            # exit(0)
    print(count)

def make_linear_fitting_one(dict_total, key_names):
    # with open('time_value_parameters.txt', 'w') as f:
    count = 0
    for key_name in key_names:
        tv_list = [*map(func_float, dict_total[key_name])]
        pd.set_option('precision',8)
        df_tv = pd.DataFrame(tv_list, columns=['time', key_name])
        df_tv.sort_values(by='time')

        # start regression
        x_col = ['time']
        y_col = [key_name]
        X = df_tv[x_col]
        y = df_tv[y_col]
        lin_reg = LinearRegression()
        model = lin_reg.fit(X, y)

        # A and b
        coef = lin_reg.coef_
        intercept = lin_reg.intercept_

        # r-square
        r_2 = model.score(X, y)
        # r_2_ = get_r_squared2(coef, intercept, X, y)

        if abs(coef[0][0]) <= 0.0001:
            count += 1
            print('X: time')
            print('y:' + key_name)
            print('a=' + str(coef), 'b=' + str(intercept))
            print('r_2: ' + str(r_2))
            print()

            sns.pairplot(df_tv, x_vars=['time'], y_vars=key_name, size=8, aspect=1.5)
            plt.show()
            # f.write('X: time' + '\n')
            # f.write('y:' + key_name + '\n')
            # f.write('a=' + str(coef) + '  b=' + str(intercept) + '\n')
            # f.write('r_2: ' + str(r_2) + '\n\n')

            # exit(0)
        print(count)



if __name__ == '__main__':

    dict_total = make_total_dict()

    # key_names in order by different types
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])



    # make_linear_fitting_pairs(dict_total, key_names)
    make_linear_fitting_one(dict_total, key_names_ordered)
