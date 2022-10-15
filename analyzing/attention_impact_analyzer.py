import argparse
import csv
import os
import pickle

import numpy as np
import pandas
from matplotlib import pyplot as plt
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from utils import extract_dataflow, indices_for_highest_attentions, process_example_for_attention_impact, \
    rank_of_the_similar_token_in_input
from tree_sitter import Language, Parser
from parser1 import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser1 import (remove_comments_and_docstrings,
                     tree_to_token_index,
                     index_to_code_token,
                     tree_to_variable_index)
from collections import Counter
import pandas as pd
import argparse
import collections
from pathlib import Path
# import pandas
# import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    # 'ruby':DFG_ruby,
    # 'go':DFG_go,
    # 'php':DFG_php,
    # 'javascript':DFG_javascript
}


def figs_for_top_3(total_counter):

    # figs_path = os.path.join(str(args.pickle_files).split('/')[0], 'figs')
    figs_path = 'figs'
    Path(figs_path).mkdir(parents=True, exist_ok=True)
    print(str(error) + ' files missing.')

    zero_rank_percentages = []
    for layer in total_counter:
        sum_val = sum(layer.values())
        zero_rank_percentages.append(((layer[0] + layer[1] + layer[2]) / sum_val) * 100)

    # total_counter_df = pandas.DataFrame(total_counter)
    # total_counter_df['sum'] = pandas.DataFrame.sum(total_counter_df, axis=1)
    # total_counter_df_new = pandas.DataFrame()
    # total_counter_df_new['1'] = total_counter_df[0]/total_counter_df['sum']
    # total_counter_df_new['2'] = total_counter_df[1]/total_counter_df['sum']
    # total_counter_df_new['3'] = total_counter_df[2]/total_counter_df['sum']
    # total_counter_df_new['4'] = total_counter_df[3]/total_counter_df['sum']
    # total_counter_df_new['5'] = total_counter_df[4]/total_counter_df['sum']
    # total_counter_df_new['3+'] = 1 - pandas.DataFrame.sum(total_counter_df_new, axis=1)
    #
    # ax = sns.heatmap(total_counter_df_new, linewidth=0.5)
    # plt.show()

    plt.bar([i + 1 for i in range(len(zero_rank_percentages))], zero_rank_percentages)
    plt.grid(axis='y')
    plt.ylim(0, 100)
    # plt.xlim(0, 10)
    plt.title("\nAppearance of similar token in top-3 attention scores for each layer\n")
    plt.xlabel('Layer number')
    plt.ylabel('Percentage %')
    plt.savefig(os.path.join(figs_path, str(args.output_fig) + '.png'))
    plt.show()
    print(zero_rank_percentages)

    # for layer_number in range(6):
    #     plt.bar(total_counter[layer_number].keys(), total_counter[5].values())
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.xlabel('Rank')
    #     plt.ylabel('Frequency')
    #     plt.savefig(os.path.join(figs_path, str(args.output_fig) + '.png'))
    #     plt.show()

    # with open('attention_impact_counter.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile, lineterminator='\n')
    #     for layer in total_counter:
    #         summ = sum(layer.values())
    #         for key, value in layer.most_common():
    #             writer.writerow([key] + [value] + [value/summ])
    #         csvfile.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_files", help="path to the pickle files that we want to sample, seperated by comma")
    parser.add_argument("--lang", help="source language", required=True)
    parser.add_argument("--model_name", help="Codebert or GraphCodebert", choices=['Codebert', 'GraphCodebert'],
                        required=True)
    parser.add_argument("--output_fig", help="name of the output figure", required=True)
    parser.add_argument("--is_CDG", help="it's code document generation", default=False)
    args = parser.parse_args()

    model_type = 'roberta'
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    _, _, tokenizer_class = MODEL_CLASSES[model_type]

    if args.model_name == 'Codebert':
        model_name_or_path = 'microsoft/codebert-base'
        tokenizer = tokenizer_class.from_pretrained('roberta-base', do_lower_case=False)
    elif args.model_name == 'GraphCodebert':
        model_name_or_path = 'microsoft/graphcodebert-base'
        tokenizer = tokenizer_class.from_pretrained('microsoft/graphcodebert-base', do_lower_case=False)
    else:
        raise ModuleNotFoundError

    # load parsers
    parsers = {}
    for lang in dfg_function:
        if os.name == 'nt':
            LANGUAGE = Language('parser1/my-languages.so', lang)
        elif os.name == 'posix':
            LANGUAGE = Language('parser1/my-languages.so', lang)
        else:
            print('os problem')
            raise ModuleNotFoundError
        parser = Parser()
        parser.set_language(LANGUAGE)
        parser = [parser, dfg_function[lang]]
        parsers[lang] = parser

    error = 0

    # total_counter = [Counter() for _ in range(6)]
    all_normalized_ranks = [[] for _ in range(6)]
    total_not_founds = 0
    files = str(args.pickle_files).split(',')
    language = args.lang
    for pickle_file in files:
        f = open(pickle_file, "rb")
        loaded_obj = pickle.load(f)
        print('model loaded')
        for idx, sample in enumerate(loaded_obj):
            try:
                ranks_coutner_for_layers, not_founds1 = rank_of_the_similar_token_in_input(sample, tokenizer)
                total_not_founds += not_founds1
                for respective_total_counter, ranks_counter_for_one_layer in zip(all_normalized_ranks, ranks_coutner_for_layers):
                    respective_total_counter += ranks_counter_for_one_layer
                    # print(respective_total_counter)
                # print(total_counter)
                if 'idx' in sample:
                    print(str(sample['idx']) + ' \\ ' + str(len(loaded_obj)) + "\n")
                else:
                    print(str(idx) + ' \\ ' + str(len(loaded_obj)) + "\n")
                # if idx > 5:
                #     break
            except Exception as e:
                print(e)
                error += 1
                pass
        f.close()
        del loaded_obj


    df = pandas.DataFrame(all_normalized_ranks)
    df['mean'] = df.mean(axis=1)

    print(total_not_founds)
    print(df['mean'])
    print('hi')

    # occurrences = [0] * 6
    # weighted_sums = [0] * 6
    # not_founds = [0] * 6
    # for idx, layer_counter in enumerate(total_counter):
    #     items = layer_counter.items()
    #     for key,val in items:
    #         if key != 10000:
    #             weighted_sums[idx] += (key + 1) * val
    #         else:
    #             not_founds[idx] += val
    #         occurrences[idx] += val
    #
    # df = pandas.DataFrame(weighted_sums)
    # df = df / occurrences[0]
    # not_found_ratio = not_founds[0] / occurrences[0]
    # df.columns = ['ranks_mean']
    # df['indices'] = df.index
    # plt.scatter(df['indices'], df['ranks_mean'], c ="blue")
    # plt.show()

    # print(occurrences)
    # print(not_founds)
    # print(weighted_sums)
    # print(df['ranks_mean'])
    # print(not_found_ratio)
    # figs_for_top_3(total_counter)
