import pre_treat as p_t
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Stable values
STABLE_VALUES = [' PayloadPowerSwitch - Channel 9 (SST Multisensor 1).Voltage.csv',
                 ' PayloadPowerSwitch - Channel 10 (SST Multisensor 2).Voltage.csv',
                 ' PowerSwitch - Channel 8 (Wing Swing BB).Voltage.csv',
                 ' PowerSwitch - Channel 10 (Wing Pitch BB).Voltage.csv',
                 ' PowerSwitch - Channel 0 (Thruster Rear BB).Voltage.csv',
                 ' PowerSwitch - Channel 9 (Wing Swing SB).Voltage.csv',
                 ' PowerSwitch - Channel 11 (Wing Pitch SB).Voltage.csv',
                 ' PowerSwitch - Channel 2 (Thruster Rear SB).Voltage.csv',
                 ' PowerSwitch - Channel 13 (Thruster SB).Voltage.csv',
                 ' PowerSwitch - Channel 7 (Jet SB).Voltage.csv',
                 ' PayloadPowerSwitch - Channel 11 (USBL).Voltage.csv',
                 ' PayloadPowerSwitch - Channel 0 (Payload CPU).Voltage.csv',
                 ' PowerSwitch - Channel 1 (Tail Fin).Voltage.csv',
                 ' PayloadPowerSwitch - Channel 6 (Videoserver).Voltage.csv',
                 ' PowerSwitch - Channel 14 (Power).Voltage.csv',
                 ' PayloadPowerSwitch - Channel 14 (Power).Voltage.csv',
                 ' PowerSwitch - Channel 5 (Payload Actor).Voltage.csv',
                 ' PayloadPowerSwitch - Channel 2 (UW CommBox).Voltage.csv',
                 ' PayloadPowerSwitch - Channel 8 (UeW CommBox).Voltage.csv',
                 ' Jet 1.Voltage.csv',
                 ' PayloadPowerSwitch - Channel 9 (SST Multisensor 1).Current.csv',
                 ' PowerSwitch - Channel 0 (Thruster Rear BB).Current.csv',
                 ' PowerSwitch - Channel 12 (Thruster BB).Current.csv',
                 ' PowerSwitch - Channel 9 (Wing Swing SB).Current.csv',
                 ' PowerSwitch - Channel 2 (Thruster Rear SB).Current.csv',
                 ' PowerSwitch - Channel 1 (Tail Fin).Current.csv',
                 ' PayloadPowerSwitch - Channel 6 (Videoserver).Current.csv',
                 ' PayloadPowerSwitch - Channel 2 (UW CommBox).Current.csv',
                 ' PayloadPowerSwitch - Channel 8 (UeW CommBox).Current.csv']

# Depth and Distance values
DD_VALUES = [' RBR Pressure.Depth.csv',
             ' Distance Chan 1.Distance.csv',
             ' Distance Chan 2.Distance.csv',
             ' Distance Chan 3.Distance.csv',
             ' Altimeter 1.Distance.csv',
             ' Altimeter 0.Distance.csv']

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

            two_entity_list = p_t.merge(first_name, second_name, dict_total)
            x_y_dict = p_t.make_data_pairs(two_entity_list)

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

            if r_2 >= 0.85:
                count += 1
                print('X:' + first_name)
                print('y:' + second_name)
                print('a=' + str(coef), 'b=' + str(intercept))
                print('r_2: ' + str(r_2))
                print()

                import seaborn as sns
                import matplotlib.pyplot as plt
                sns.pairplot(df, x_vars=first_name, y_vars=second_name , size=9, aspect=1.5, kind='reg')
                # sns.pairplot(df, vars=[first_name, second_name], size=6, aspect=2)
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
    # import csv
    # with open('time_value_parameters.csv', 'w') as f:
    #     csv_writer = csv.writer(f, dialect='excel')
    #     csv_writer.writerow(['X(time compressed)', 'y', 'r^2', 'A', 'b'])
        count = 0
        for key_name in key_names:
            tv_list = dict_total[key_name]
            pd.set_option('precision',8)
            df_tv = pd.DataFrame(tv_list, columns=['time', key_name])
            df_tv.sort_values(by='time')

            # start regression
            x_col = 'time'
            y_col = key_name
            X = np.array(df_tv[x_col])
            y = np.array(df_tv[y_col])

            # make value of time smaller
            X_delta_sp = X[0] / 100
            # X_cms = [*map(lambda a: a/1000-X_delta_sp, X)]
            X_cms = X / 100 - X_delta_sp
            X_cms = X_cms.reshape(X_cms.shape[0], 1)
            lin_reg = LinearRegression()
            model = lin_reg.fit(X_cms, y)

            # A and b
            coef = lin_reg.coef_
            intercept = lin_reg.intercept_

            # r-square
            r_2 = model.score(X_cms, y)
            # r_2_ = get_r_squared2(coef, intercept, X, y)

            if abs(coef[0]) >= 0:
                count += 1
                print('X: time')
                print('y:' + key_name)
                print('a=' + str(coef), 'b=' + str(intercept))
                print('r_2: ' + str(r_2))
                print()

                import matplotlib.pyplot as plt
                plt.figure(figsize=(16,9))
                plt.scatter(X_cms, y, color='red', label='Sample Point', linewidths=0.5)
                plt.xlabel('time')
                plt.ylabel(key_name)
                plt.plot(X_cms, lin_reg.predict(X_cms), color='orange', label='Linear', linewidth=3)
                plt.show()
                # plt.savefig('stable_values_' + str(count) + '.png', dpi=200)

                # f.write('X: time compressed' + '\n')
                # f.write('y:' + key_name + '\n')
                # f.write('a=' + str(coef) + '  b=' + str(intercept) + '\n')
                # f.write('r_2: ' + str(r_2) + '\n\n')

                # write to a csv
                # csv_writer.writerow(['time', key_name, r_2, coef[0], intercept])


                # exit(0)
            print(count)

if __name__ == '__main__':

    dict_total = p_t.make_total_dict()

    # key_names in order by different types
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])
    key_names_ordered_Stable = [key_name for key_name in key_names_ordered if key_name in STABLE_VALUES]


    # make_linear_fitting_pairs(dict_total, key_names_ordered)
    make_linear_fitting_one(dict_total, key_names_ordered_Stable)