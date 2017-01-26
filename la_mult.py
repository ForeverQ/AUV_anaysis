import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import src.linear_fit as l_f
import src.linear_analysis as l_a


FILE_PATH = './mult_dm/'
# FILE_PATH = './'

def func_float(a_list):
    a_list[0] = float(a_list[0])
    a_list[2] = float(a_list[2])
    return [a_list[0],a_list[2]]


#
# X and y are pd.DataFrame
# coef is np.array, intercept is np.float64
def get_r_squared2(coef , intercept, X, y):


    X_np = X.as_matrix()
    y_np = y.as_matrix()

    y_bar = np.sum(y_np) / len(y_np)
    ss_t = np.sum(list(map(lambda a,b: (a-b)**2, y_np,[y_bar]*len(y_np))))
    Y_np = np.array(list(map(lambda a: np.dot(coef.T, a) + intercept, X_np)))
    ss_r = np.sum(list(map(lambda a,b: (a-b)**2, Y_np, [y_bar]*len(y_np))))
    return ss_r / ss_t

if __name__ == '__main__':

    file_names = l_a.generate_csv_file_names(file_path=FILE_PATH)
    data_merge = DataFrame()
    for file_name in file_names:
        data = pd.read_csv(FILE_PATH + file_name)
        if data_merge.size == 0:
            data_merge = data
        else:
            # TODO may have some detail problems of merge in the future
            data_merge = pd.merge(data_merge, data)
    # data_merge.rename(columns={'timestamp (seconds since 01/01/1970)': 'timestamp',
    #                            ' x (m/s/s)': 'x.Acc', ' y (m/s/s)': 'y.Acc', ' z (m/s/s)': 'z.Acc',
    #                            ' x (rad/s)': 'x.Ang', ' y (rad/s)': 'y.Ang', ' z (rad/s)': 'z.Ang',
    #                            ' phi (rad)': 'phi.Eul', ' theta (rad)': 'theta.Eul',
    #                            ' psi (rad)': 'psi.Eul', ' psi_magnetic (rad)': 'psi_magnetic.Eul'},
    #                   inplace=True)


    # TODO use list instead of np
    data_sp = data_merge
    data_sp_list = list(data_sp.as_matrix())

    # about one dimension data
    dict_total = l_a.make_total_dict()
    key_names = sorted(dict_total, key=lambda item: item[::-1])
    for key_name in key_names:
        data_sm = dict_total[key_name]
        data_sm_list = [*map(list, data_sm)]
        data_sm_list = [*map(func_float, data_sm_list)]

        new_data_sp_list = []
        for item_sg in data_sm_list:
            for j in range(len(data_sp_list)-1):
                diff_curr = abs(item_sg[0]-data_sp_list[j][0])
                diff_next = abs(item_sg[0]-data_sp_list[j+1][0])

                if ((diff_curr) >= 1):
                    continue
                else:
                    if (diff_next >= diff_curr):
                        new_data_sp_list.append(data_sp_list[j])
                        break
                    else:
                        continue
        new_data_sp_df = DataFrame(new_data_sp_list)
        data_sm_df = DataFrame(data_sm_list)
        new_data_sp_df['value of ' + key_name] = data_sm_df[1]
        new_data_sp_df.rename(columns={0: 'timestamp',
                               4: 'x.Acc', 5: 'y.Acc', 6: 'z.Acc',
                               7: 'x.Ang', 8: 'y.Ang', 9: 'z.Ang',
                               10: 'phi.Eul', 11: 'theta.Eul',
                               12: 'psi.Eul', 13: 'psi_magnetic.Eul'},
                      inplace=True)

        # start regression
        feature_cols = ['x.Acc', 'y.Acc', 'z.Acc', 'x.Ang', 'y.Ang', 'z.Ang',
                        'phi.Eul', 'theta.Eul', 'psi.Eul', 'psi_magnetic.Eul']
        X = new_data_sp_df[feature_cols]
        y = new_data_sp_df['value of ' + key_name]
        lin_reg = LinearRegression()
        model = lin_reg.fit(X, y)

        # A and b
        coef = lin_reg.coef_
        intercept = lin_reg.intercept_

        # r-square
        r_2 = get_r_squared2(coef, intercept, X, y)
        r_2_ = model.score(X, y)
        if r_2 >= 0.5:
            print(coef, intercept)
            print('r_2: ' + str(r_2))
            print('r_2_: ' + str(r_2_))
            sns.pairplot(new_data_sp_df, x_vars=['x.Acc','y.Acc','z.Acc',
                                                 'x.Ang','y.Ang','z.Ang',
                                                 'phi.Eul','theta.Eul','psi.Eul', 'psi_magnetic.Eul'], y_vars='value of ' + key_name , size=8, aspect=0.8, kind='reg')
            plt.show()
            # plt.savefig('picture_' + key_name + '.png', dpi=150)

            # exit(0)

















