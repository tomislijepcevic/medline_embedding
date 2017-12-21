def get_inliers(X, outliers_fraction=0.02):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(contamination=outliers_fraction)
    is_inlier = clf.fit(X).predict(X) == 1

    return is_inlier


def plot(ax, df):
    X = df[['0', '1']].values
    is_inlier = get_inliers(X)
    df = df[is_inlier]

    from sklearn.metrics import silhouette_score

    X = df[['0', '1']].values
    y = df['y'].values
    s = silhouette_score(X, y)

    for c, g in df.groupby('y'):
        ax.scatter(g['0'], g['1'], label=c)

    ax.set_xlabel('silhueta: %.2f' % s, fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(
        axis='y',
        which='both',
        left='off',
        right='off',
        labelleft='off')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('points')
    parser.add_argument('dest')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(args.points)
    fig, ax = plt.subplots()
    plot(ax, df)
    handles = ax.get_legend_handles_labels()[0]
    plt.legend(handles=handles, loc="upper center", ncol=3,
               shadow=True, fancybox=True)
    fig.savefig(args.dest)


if __name__ == '__main__':
    main()
