import csv
from itertools import groupby

with open('Depth.csv') as f:
    f_csv = csv.reader(f)
    depth_data_lists = list(f_csv)
    depth_data_tuples = []
    for row in depth_data_lists[1:]:
        depth_data_tuples.append((row[0], row[2], row[3]))

    # use this method to classify different entities
    depth_data_tuples_entity_classify = sorted(depth_data_tuples, key=lambda item: item[1])
    depth_dic_by_entity = dict()
    for entity, tuples in groupby(depth_data_tuples_entity_classify, lambda item:item[1]):
        depth_dic_by_entity[entity] = list(tuples)


with open('Current.csv') as f:
    f_csv = csv.reader(f)
    current_data_lists = list(f_csv)
    current_data_tuples = []
    for row in current_data_lists[1:]:
        current_data_tuples.append((row[0], row[2], row[3]))
    current_data_tuples_entity_classify = sorted(current_data_tuples, key=lambda item: item[1])
    current_dict_by_entity = dict()
    for entity, tuples in groupby(current_data_tuples_entity_classify, lambda item:item[1]):
        current_dict_by_entity[entity] = list(tuples)
    # print(len(current_dict_by_entity))


two_entities = depth_dic_by_entity[' RBR Pressure'] + current_dict_by_entity[' PayloadPowerSwitch - Channel 0 (Payload CPU)']
two_entities_sorted_by_time = sorted(two_entities, key=lambda item: item[0])

with open('Depth&Current.out', 'w') as f:
    for tuple in two_entities_sorted_by_time:
        f.write(str(tuple) + '\n')



    # current_different_entities = []
    # for i in range(len(current_data_tuples_entity_classify)-1):
    #     if current_data_tuples_entity_classify[i] != current_data_tuples_entity_classify[i+1]:
    #         current_different_entities.append(current_data_tuples_entity_classify[])