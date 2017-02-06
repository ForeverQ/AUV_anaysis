import pre_process as p_p
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

TIMESTAMP_RATIO = 100
VALUE_PRECISION = 8
RS_THRESHOLD = 0.85
SLOPE_THRESHOLD = 0.0001


def make_linear_fitting_pairs(dict_total, key_names):
    """Combines each of two data values and conducts linear regression on them
       Uses a threshold of R^2 to filter data pairs which have better result of linear regression
    """
    count = 0
    for i in range(0,len(key_names)):
        for j in range(0,len(key_names)):
            if key_names[i] == key_names[j]:
                continue
            first_name = key_names[i]
            second_name = key_names[j]
            two_entity_list = p_p.merge(first_name, second_name, dict_total)
            x_y_dict = p_p.make_data_pairs(two_entity_list)
            pd.set_option('precision', VALUE_PRECISION)
            df = pd.DataFrame(x_y_dict)
            df.sort_values(by='time')

            # Starts regression
            x_col = [first_name]
            y_col = [second_name]
            X = df[x_col]
            y = df[y_col]
            lin_reg = LinearRegression()
            model = lin_reg.fit(X, y)

            # A and b
            coef = lin_reg.coef_
            intercept = lin_reg.intercept_

            # R^2
            r_2 = model.score(X, y)
            if r_2 >= RS_THRESHOLD:
                count += 1
                print('X:' + first_name)
                print('y:' + second_name)
                print('a=' + str(coef), 'b=' + str(intercept))
                print('r_2: ' + str(r_2))
                print()

                # Starts drawing
                import seaborn as sns
                import matplotlib.pyplot as plt
                sns.pairplot(df, x_vars=first_name, y_vars=second_name , size=9, aspect=1.5, kind='reg')
                # sns.pairplot(df, vars=[first_name, second_name], size=6, aspect=2)
                plt.show()
                # plt.savefig(first_name + ' && ' + second_name + '.png', dpi=150)
    print(count)

def make_linear_fitting_one(dict_total, key_names):
    """Conducts linear regression on one kind of data value.
       Uses a threshold of slope which is close to 0 to filter values which are supposed to be "Stable"
    """
    count = 0
    for key_name in key_names:
        tv_list = dict_total[key_name]
        pd.set_option('precision',8)
        df_tv = pd.DataFrame(tv_list, columns=['time', key_name])
        df_tv.sort_values(by='time')

        # Starts regression
        x_col = 'time'
        y_col = key_name
        X = np.array(df_tv[x_col])
        y = np.array(df_tv[y_col])
        X_delta_sp = X[0] / TIMESTAMP_RATIO
        X_cms = X / TIMESTAMP_RATIO - X_delta_sp
        X_cms = X_cms.reshape(X_cms.shape[0], 1)
        lin_reg = LinearRegression()
        model = lin_reg.fit(X_cms, y)

        # A and b
        coef = lin_reg.coef_
        intercept = lin_reg.intercept_

        # R^2
        r_2 = model.score(X_cms, y)
        if abs(coef[0]) <= SLOPE_THRESHOLD:
            count += 1
            print('X: time')
            print('y:' + key_name)
            print('a=' + str(coef), 'b=' + str(intercept))
            print('r_2: ' + str(r_2))
            print()

            # Starts drawing
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16,9))
            plt.scatter(X_cms, y, color='red', label='Sample Point', linewidths=0.5)
            plt.xlabel('time')
            plt.ylabel(key_name)
            plt.plot(X_cms, lin_reg.predict(X_cms), color='orange', label='Linear', linewidth=3)
            plt.show()
            # plt.savefig('stable_values_' + str(count) + '.png', dpi=200)
    print(count)

if __name__ == '__main__':
    dict_total = p_p.make_total_dict()
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])
    key_names_ordered_Stable = [key_name for key_name in key_names_ordered if key_name in STABLE_VALUES]

    # make_linear_fitting_pairs(dict_total, key_names_ordered)
    make_linear_fitting_one(dict_total, key_names_ordered)