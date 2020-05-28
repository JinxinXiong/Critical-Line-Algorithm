# ---------------------------------------------------------------
def plot2D(x, y, xLabel='', yLabel='', title='', pathChart=None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # one row, one column, first plot
    ax.plot(x, y, color='blue')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel, rotation=90)
    plt.xticks(rotation='vertical')
    plt.title(title)
    if pathChart is None:
        plt.show()
    else:
        plt.savefig(pathChart)
    plt.show()
    return


# ---------------------------------------------------------------
def main():
    import numpy as np
    import CLA
    # 1) Path
    path = 'CLA_Data.csv'
    # 2) Load data, set seed
    headers = open(path, 'r').readline()[:-1].split(',')
    data = np.genfromtxt(path, delimiter=',', skip_header=1)  # load as numpy array
    mean = np.array(data[:1]).T
    lB = np.array(data[1:2]).T
    uB = np.array(data[2:3]).T
    covar = np.array(data[3:])
    # 3) Invoke object
    cla = CLA.CLA(mean, covar, lB, uB)
    cla.solve()
    print('===============All Turning point====================')
    for w in cla.w:
        print(w)
    # print(cla.w)  # print all turning points
    # 4) Plot frontier
    mu, sigma, weights = cla.efFrontier(500)
    # plot2D(sigma, mu, 'Risk', 'Expected Excess Return', 'CLA-derived Efficient Frontier', '../')
    # 5) Get Maximum Sharpe ratio portfolio
    print("==============Get Maximum Sharpe ratio portfolio===============")
    sr, w_sr = cla.getMaxSR()
    print(np.dot(np.dot(w_sr.T, cla.covar), w_sr)[0, 0] ** .5, sr)
    print(w_sr)
    print(sum(w_sr))
    # 6) Get Minimum Variance portfolio
    print("==============Get Minimum Variance portfolio=================")
    mv, w_mv = cla.getMinVar()
    print(mv)
    print(w_mv)
    print(sum(w_mv))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # one row, one column, first plot
    ax.plot(sigma, mu, color='blue', label='Efficient Frontier')
    ax.set_xlabel('Risk')
    ax.set_ylabel('Expected Excess Return', rotation=90)
    plt.xticks(rotation='vertical')
    plt.title('CLA-derived Efficient Frontier')
    sr = ax.scatter(np.dot(np.dot(w_sr.T, cla.covar), w_sr)[0, 0] ** .5, np.dot(w_sr.T, cla.mean)[0, 0])
    mv = ax.scatter(np.dot(np.dot(w_mv.T, cla.covar), w_mv)[0, 0] ** .5, np.dot(w_mv.T, cla.mean)[0, 0])
    plt.legend((sr, mv), ('maximum sharpe ratio', 'minimum variance'))
    plt.show()
    return


# ---------------------------------------------------------------
# Boilerplate
if __name__ == '__main__':
    main()
