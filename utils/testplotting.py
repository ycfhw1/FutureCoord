import os
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/home/xiaofu/FutureCoord/resultsdemo')
    parser.add_argument('--output', type=str, default='./results')
    args = parser.parse_args()

    index_mapping = {'agent':'agent','episode': 'Episode'}

    measure_mapping = {'acceptance_rate': 'accept_rate','mean_resd_latency':'mean_resd_latency'}

    results = pd.DataFrame()

    dirs = [directory for directory in os.listdir(args.logdir)]
    tables = [Path(args.logdir) /directory / 'results'/
              'results.csv' for directory in dirs]
    tables = [table for table in tables if table.exists()]
    print(tables)
    for table in tables:
        data = pd.read_csv(table)
        results = pd.concat((results, data))

    results = results.rename(columns={**index_mapping, **measure_mapping})
    results = results.reset_index()
    sns.set_style("whitegrid")
    for measure in measure_mapping.values():
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x='agent', y=measure, data=results, ax=ax)
        sns.despine()
        fig.savefig(Path(args.output) / f'{measure}_agent.pdf')
        fig.savefig(Path(args.output) / f'{measure}_agent.pdf')
        # fig, ax = plt.subplots(figsize=(5, 5))
        # sns.lineplot(x="Episode", y=measure,hue="agent", style="agent", data=results)
        # sns.despine()
        # fig.savefig(Path(args.output) / f'{measure}_agent.pdf')