import argparse
import math
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot
from pathlib import Path


# def plot_the_feature(data, label, bins, i):
#     ##computing the bin properties (same for both distributions)
#     num_bin = bins
#     # bin_lims = np.linspace(0, 1, num_bin + 1)
#     bin_lims = np.linspace(data[label].min(), data[label].max(), num_bin + 1)
#     bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
#     bin_widths = bin_lims[1:] - bin_lims[:-1]
#
#     hist1, _ = np.histogram(data[label], bins=bins)
#     hist2, _ = np.histogram(nearest_tenth[label], bins=bins)
#     ##normalizing
#     hist1b = hist1 / np.max(hist1)
#     hist2b = hist2 / np.max(hist2)
#
#     # pyplot.hist([hist1b, hist2b], 20, alpha=0.5, label=['all','easy-bad'], color=['#7F7FFF', 'r'])
#
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
#
#     ax1.bar(bin_centers, hist1, width=bin_widths, align='center')
#     ax1.bar(bin_centers, hist2, width=bin_widths, align='center', alpha=0.5)
#     ax1.set_title('original')
#
#     ax2.bar(bin_centers, hist1b, width=bin_widths, align='center')
#     ax2.bar(bin_centers, hist2b, width=bin_widths, align='center', alpha=0.5)
#     ax2.set_title(str(label + '-normalized'))
#
#     plt.title(label)
#     plt.xlabel(label)
#     plt.ylabel('frequency')
#     pyplot.legend(loc='upper right')
#     plt.savefig('_figs/RQ3_plot_' + str(i) + '.png')
#     pyplot.show()

def plot_the_feature(data, label, bins, i):

    hist1, _ = np.histogram(data[label], bins=bins, range=[data[label].min(), data[label].max()])
    hist2, _ = np.histogram(nearest_tenth[label], bins=bins, range=[data[label].min(), data[label].max()])
    ##normalizing
    hist1b = hist1 / np.sum(hist1) * 100
    hist2b = hist2 / np.sum(hist2) * 100

    ##computing the bin properties (same for both distributions)
    num_bin = bins

    bin_lims1 = np.linspace(data[label].min(), data[label].max(), num_bin + 1)
    bin_centers1 = 1 * (bin_lims1[:-1] + bin_lims1[1:])/2
    bin_widths1 = bin_lims1[1:] - bin_lims1[:-1]

    # bin_lims2 = np.linspace(nearest_tenth[label].min(), nearest_tenth[label].max(), num_bin + 1)
    # bin_centers2 = (bin_lims2[:-1] + bin_lims2[:-1])/2
    # bin_widths2 = bin_lims2[1:] - bin_lims2[:-1]

    plt.bar(bin_centers1, hist1b, width=bin_widths1, align='center')
    plt.bar(bin_centers1, hist2b, width=bin_widths1, align='center', alpha=0.5, color='red')
    # plt.title(str(label.replace('_', ' ')))

    # plt.title(label)
    xlabel = ''
    if label == 'distances':
        xlabel = 'levenshtein distance'
    elif label == 'n_tokens':
        xlabel = 'number of tokens'
    elif label == 'cyclomatic_complexity':
        xlabel = 'cyclomatic complexity'
    elif label == 'nested_block_depth':
        xlabel = 'nested block depth'
    elif label == 'variable':
        xlabel = 'number of variables'
    elif label == 'CDG_overlap':
        xlabel = 'overlap'
    plt.xlabel(xlabel)
    plt.ylabel('Percentage')
    # pyplot.legend(loc='upper right')
    plt.savefig('RQ3_figs/RQ3_plot_' + str(i) + '_' + label + '.png', bbox_inches='tight')
    pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_files", help="path to the pickle files that we want to sample, seperated by comma")
    parser.add_argument("--lang", help="source language", required=True)
    parser.add_argument("--model_name", help="Codebert or GraphCodebert", choices=['Codebert', 'GraphCodebert'],
                        required=True)
    args = parser.parse_args()
    language = args.lang
    files = str(args.pickle_files).split(',')

    print("now =", datetime.now())

    BLEUS_TRESHOLD = 0.33
    i = 0
    populations = pandas.DataFrame()
    for pickle_file in files:
        distances_from_zero = []
        f = open(pickle_file, "rb")
        loaded_obj = pickle.load(f)
        data = pd.DataFrame(loaded_obj)
        # data['lev_distances_normalized'] = (data['distances'] - data['distances'].min()) / (
        #         data['distances'].max() - data['distances'].min())
        # for lev_dist, bleu in zip(data['lev_distances_normalized'], data['bleus']):
        #     distances_from_zero.append(math.dist([lev_dist, bleu], [0, 0]))
        # data['distances_from_zero'] = distances_from_zero

        # nearest_tenth = data.sort_values(by=['distances_from_zero'])[0:round(len(distances_from_zero) / 10)]
        # nearest_tenth = data.sort_values(by=['bleus'])[0:round(len(data['bleus']) / 10)]

        is_CDG = not (('CDG_overlap' not in data) or (data.head(1)['CDG_overlap'] is None))
        if not is_CDG:
            first_third_bleus = data.sort_values('bleus')[:int(len(data) / 3)]
            first_third_distance = data.sort_values('distances')[:int(len(data) / 3)]
            nearest_tenth = pd.merge(first_third_bleus, first_third_distance, how='inner', on=[x for x in list(data.columns)])
        else:
            first_third_bleus = data.sort_values('bleus')[:int(len(data) / 3)]
            first_third_overlaps = data.sort_values('CDG_overlap')[-int(len(data) / 3):]
            nearest_tenth = pd.merge(first_third_bleus, first_third_overlaps, how='inner', on=[x for x in list(data.columns)])

        populations[pickle_file.split('/')[0]] = [len(data), len(nearest_tenth), len(nearest_tenth)/len(data)]
        # nearest_tenth = data[data['bleus'] < BLEUS_TRESHOLD]
        # nearest_tenth = nearest_tenth[nearest_tenth['distances'] < 0.33 * data['distances'].max()]

        ### only nearest tenth scatters ###
        # nearest_tenth.hist(column=['distances'],bins=50)
        # plt.show()
        # # nearest_tenth.plot.scatter(x='distances', y='bleus', c='red', s=10)
        # # nearest_tenth.plot.scatter(x='n_tokens', y='bleus', c='red', s=10)
        # # nearest_tenth.plot.scatter(x='cyclomatic_complexity', y='bleus', c='red', s=10)
        # # nearest_tenth.plot.scatter(x='nested_block_depth', y='bleus', c='red', s=10)
        # # nearest_tenth.plot.scatter(x='variable', y='bleus', c='red', s=10)
        # # plt.show()

        if is_CDG:
            ### all and nearest tenth scatters for CDG ###
            ax = data.plot.scatter(x='CDG_overlap', y='bleus', c='blue', s=10)
            nearest_tenth.plot.scatter(x='CDG_overlap', y='bleus', c='red', s=10, ax=ax)
        else:
            ### all and nearest tenth scatters ###
            ax = data.plot.scatter(x='distances', y='bleus', c='blue', s=10)
            nearest_tenth.plot.scatter(x='distances', y='bleus', c='red', s=10, ax=ax)

        figs_path = 'RQ3_figs'
        Path(figs_path).mkdir(parents=True, exist_ok=True)

        plt.savefig(os.path.join(figs_path, ('RQ3_plot_' + str(i) + '_population.png')), bbox_inches='tight')
        i += 1
        plt.show()
        # ax = data.plot.scatter(x='n_tokens', y='bleus', c='blue', s=10)
        # nearest_tenth.plot.scatter(x='n_tokens', y='bleus', c='red', s=10, ax=ax)
        # plt.show()
        # ax = data.plot.scatter(x='cyclomatic_complexity', y='bleus', c='blue', s=10)
        # nearest_tenth.plot.scatter(x='cyclomatic_complexity', y='bleus', c='red', s=10, ax=ax)
        # plt.show()
        # ax = data.plot.scatter(x='nested_block_depth', y='bleus', c='blue', s=10)
        # nearest_tenth.plot.scatter(x='nested_block_depth', y='bleus', c='red', s=10, ax=ax)
        # plt.show()
        # ax = data.plot.scatter(x='variable', y='bleus', c='blue', s=10)
        # nearest_tenth.plot.scatter(x='variable', y='bleus', c='red', s=10, ax=ax)
        # plt.show()

        ### bar plots for all and nearest tenth ###
        if is_CDG:
            labels_to_plot = ['CDG_overlap', 'n_tokens', 'cyclomatic_complexity', 'nested_block_depth', 'variable']
        else:
            labels_to_plot = ['distances', 'n_tokens', 'cyclomatic_complexity', 'nested_block_depth', 'variable']
        for label in labels_to_plot:
            plot_the_feature(data=data, label=label, bins=40, i=i)
            i += 1

        nearest_tenth.to_csv(
            str(pickle_file.split('/')[0]) + '_nearest_tenth_indices_under{}.csv'.format(str(BLEUS_TRESHOLD * 100)),
            header=None,
            columns=['index'], index=False, )

        f.close()
        del loaded_obj

    writer = pd.ExcelWriter('easy-bad-population.xlsx', engine='xlsxwriter')
    populations.to_excel(writer, sheet_name='easy-bad-population')
    writer.save()