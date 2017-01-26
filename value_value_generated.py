
import numpy as np

def __f_3(t):
    return np.exp(-t) * np.cos(2*np.pi*t), 'exp and cos'

def __f_1(t):
    return np.exp(t), 'exp'

def __rec_f_0(t):
    y = []
    value = 1
    for item in t:
        if ((item+0.5*np.pi)/(2*np.pi))//1%2 == 0:
            y.append(-1*value)
        else:
            y.append(1*value)
    return np.array(y), 'rectangle'

def sin_f_0(t):
    return np.sin(t), 'sin'

def sin_f_1(t):
    return np.sin(t/(19/4)), 'sin (1T)'

def cos_f_0(t):
    return np.cos(t), 'cos'

def cos_f_1(t):
    return np.cos(t/(19/4)), 'cos (1T)'

def tri_fSin_f_0(t):
    y = []
    for item in t:
        item_m = (item-0.5*np.pi)/np.pi
        if item_m//1%2 == 0:
            x_m = item_m % 1 + 0
            y.append(-1*x_m+1)
        else:
            x_m = item_m % 1 + 1
            y.append(1*x_m-1)
    return np.array(y), 'triangle for sin'

def tri_fSin_f_1(t):
    y = []
    for item in (t/(19/4)):
        item_m = (item-0.5*np.pi)/np.pi
        if item_m//1%2 == 0:
            x_m = item_m % 1 + 0
            y.append(-1*x_m+1)
        else:
            x_m = item_m % 1 + 1
            y.append(1*x_m-1)
    return np.array(y), 'triangle for sin (1T)'

def tri_fCos_f_0(t):
    y = []
    for item in t:
        item_m = (item)/np.pi
        if item_m//1%2 == 0:
            x_m = item_m % 1 + 0
            y.append(-1*x_m+1)
        else:
            x_m = item_m % 1 + 1
            y.append(1*x_m-1)
    return np.array(y), 'triangle for cos'

def tri_fCos_f_1(t):
    y = []
    for item in (t/(19/4)):
        item_m = (item)/np.pi
        if item_m//1%2 == 0:
            x_m = item_m % 1 + 0
            y.append(-1*x_m+1)
        else:
            x_m = item_m % 1 + 1
            y.append(1*x_m-1)
    return np.array(y), 'triangle for cos (1T)'

def constant_f_0(t):
    return np.linspace(1,1,len(t)), 'constant value of 1'

def upSlopeLinear_f_0(t):
    return 2*t+3, 'up slope linear func'

def downSlopeLinear_f_0(t):
    return -3*t-1, 'down slope linear func'

def calculate_derivative_series(series_X, series_y):
    series_X_d = []
    series_y_d = []
    for (index_X, item_X), (index_y, item_y) in zip(enumerate(series_X), enumerate(series_y)):
        if index_X >= len(series_X) - 1:
            break
        series_X_d.append((series_X[index_X+1] + series_X[index_X])/2.0)
        series_y_d.append((series_y[index_y+1] - series_y[index_y])/(series_X[index_X+1] - series_X[index_X]))
    return np.array(series_X_d), np.array(series_y_d)

def draw_two(t, y_1, name_1, y_2, name_2):
    series_y_1_d, series_y_2_d = calculate_derivative_series(y_1, y_2)
    series_y_1_d_range = np.amax(series_y_1_d) - np.amin(series_y_1_d)
    series_y_1_d_max, series_y_1_d_min = (np.amax(series_y_1_d)+0.08*series_y_1_d_range,
                                          np.amin(series_y_1_d)-0.08*series_y_1_d_range)
    series_y_2_d_range = np.amax(series_y_2_d) - np.amin(series_y_2_d)
    series_y_2_d_max, series_y_2_d_min = (np.amax(series_y_2_d)+0.08*series_y_2_d_range,
                                          np.amin(series_y_2_d)-0.08*series_y_2_d_range)

    t_range = np.amax(t) - np.amin(t)
    t_max, t_min = (np.amax(t)+0.08*t_range, np.amin(t)-0.08*t_range)
    y_1_range = np.amax(y_1) - np.amin(y_1)
    y_1_max, y_1_min = (np.amax(y_1)+0.08*y_1_range, np.amin(y_1)-0.08*y_1_range)
    y_2_range = np.amax(y_2) - np.amin(y_2)
    y_2_max, y_2_min = (np.amax(y_2)+0.08*y_2_range, np.amin(y_2)-0.08*y_2_range)

    import matplotlib.pyplot as plt

    plt.figure(1, figsize=(8,8))
    plt.subplot(211)
    plt.title(name_1)
    plt.plot(t, y_1, color='red', label='', linewidth=2)
    plt.axis([t_min, t_max, y_1_min, y_1_max])
    plt.subplot(212)
    plt.title(name_2)
    plt.plot(t, y_2, color='blue', label='', linewidth=2)
    plt.axis([t_min, t_max, y_2_min, y_2_max])
    # plt.savefig(name_1 + ' && ' + name_2 + '.png', dpi=200)
    # plt.close()

    # combinations of simple version
    # plt.figure(2, figsize=(8,8))
    # plt.subplot(211)
    # plt.title(name_1 + '&' + name_2)
    # plt.plot(y_1, y_2, color='orange', label='', linewidth=2)
    # plt.axis([y_1_min, y_1_max, y_2_min, y_2_max])
    # plt.subplot(212)
    # plt.title('derivative of ' + name_1 + '&' + name_2)
    # plt.plot(series_y_1_d, series_y_2_d, color='c', label='', linewidth=2)
    # plt.axis([series_y_1_d_min, series_y_1_d_max,series_y_2_d_min, series_y_2_d_max])

    # plt.figure(3, figsize=(16,9))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))
    cm = plt.cm.get_cmap('rainbow')
    z_3_1 = np.arange(1000)
    # ax1.set_title(name_1 + '&' + name_2)
    ax1.set_xlabel(name_1)
    ax1.set_ylabel(name_2)
    ax1.scatter(y_1, y_2, c=z_3_1, s=100, cmap=cm, label='', alpha=.6)
    ax1.axis([y_1_min, y_1_max, y_2_min, y_2_max])
    z_3_2 = np.arange(999)
    # ax2.set_title('derivative of ' + name_1 + '&' + name_2)
    ax2.set_xlabel(name_1)
    ax2.set_ylabel('derivative of ' + name_2)
    sc_3_2 = ax2.scatter(series_y_1_d, series_y_2_d, c=z_3_2, s=100, cmap=cm, label='', alpha=.6)
    ax2.axis([series_y_1_d_min, series_y_1_d_max,series_y_2_d_min, series_y_2_d_max])
    plt.colorbar(sc_3_2)
    # plt.savefig(name_1 + ' && ' + name_2 + ' combination'+ '.png', dpi=200)
    plt.show()

def generate_two_funcs(func_first, func_second):
    t = np.linspace(0, 30, 1000)
    y_1, name_1 = func_first(t)
    y_2, name_2 = func_second(t)
    return t, y_1,name_1, y_2, name_2

if __name__ == '__main__':

    # draw_two(*generate_two_funcs(eval('sin_f_'+str(0)), eval('downSlopeLinear_f_'+str(0))))
    # draw_two(*generate_two_funcs(eval('downSlopeLinear_f_'+str(0)), eval('sin_f_'+str(0))))
    #
    # draw_two(*generate_two_funcs(eval('sin_f_'+str(0)), eval('constant_f_'+str(0))))
    # draw_two(*generate_two_funcs(eval('constant_f_'+str(0)), eval('sin_f_'+str(0))))
    #
    # draw_two(*generate_two_funcs(eval('upSlopeLinear_f_'+str(0)), eval('downSlopeLinear_f_'+str(0))))
    # draw_two(*generate_two_funcs(eval('downSlopeLinear_f_'+str(0)), eval('upSlopeLinear_f_'+str(0))))

    draw_two(*generate_two_funcs(eval('cos_f_'+str(1)), eval('tri_fCos_f_'+str(1))))
    draw_two(*generate_two_funcs(eval('tri_fCos_f_'+str(1)), eval('cos_f_'+str(1))))

    draw_two(*generate_two_funcs(eval('cos_f_'+str(0)), eval('tri_fCos_f_'+str(0))))
    draw_two(*generate_two_funcs(eval('tri_fCos_f_'+str(0)), eval('cos_f_'+str(0))))

    draw_two(*generate_two_funcs(eval('sin_f_'+str(1)), eval('tri_fSin_f_'+str(1))))
    draw_two(*generate_two_funcs(eval('tri_fSin_f_'+str(1)), eval('sin_f_'+str(1))))

    draw_two(*generate_two_funcs(eval('sin_f_'+str(0)), eval('tri_fSin_f_'+str(0))))
    draw_two(*generate_two_funcs(eval('tri_fSin_f_'+str(0)), eval('sin_f_'+str(0))))



