import collections
import csv
import difflib
import os
import pickle

import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from utils import extract_dataflow, indices_for_highest_attentions, FIXED_KEYWORDS, sum_of_attention_for_tokens
from tree_sitter import Language, Parser
from parser1 import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser1 import (remove_comments_and_docstrings,
                     tree_to_token_index,
                     index_to_code_token,
                     tree_to_variable_index)
from collections import Counter
import pandas as pd
import argparse
from datetime import datetime

# pickle_file = 'saved_model/prediction_pickle(TEST).pickle'

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    # 'ruby':DFG_ruby,
    # 'go':DFG_go,
    # 'php':DFG_php,
    # 'javascript':DFG_javascript
}


def process_example(example, parsers, lang):
    ##extract data flow
    code_tokens, dfg, types, important_tokens = extract_dataflow(example['source'], parsers[lang], lang)
    # code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
    #                enumerate(code_tokens)]
    code_tokens_tokenized = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                             enumerate(code_tokens)]
    # code_tokens_tokenized = [tokenizer.tokenize(x) for x in code_tokens]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    cur_pos2ori = {}
    for i in range(len(code_tokens_tokenized)):
        start_pos = ori2cur_pos[i - 1][1]
        end_pos = ori2cur_pos[i - 1][1] + len(code_tokens_tokenized[i])
        ori2cur_pos[i] = (start_pos, end_pos)
        for j in range(start_pos, end_pos):
            cur_pos2ori[j] = i
    code_tokens_tokenized = [y for x in code_tokens_tokenized for y in x]
    return code_tokens, code_tokens_tokenized, ori2cur_pos, cur_pos2ori, types, important_tokens


# def colorize_in_html(sample):
#     f = open('colorize/' + str(sample['idx']) + '.html', 'w', encoding='utf8')
#     f.write('source:')
#     f.write('<br>')
#     f.write(sample['source'])
#     f.write('<br>')
#     f.write('target:')
#     f.write('<br>')
#     f.write(sample['target'])
#     f.write('<br>')
#     f.write('best prediction:')
#     f.write('<br>')
#     f.write(sample['prediction'])
#     f.write('<br>')
#     f.write('<br><br><br>')
#
#     code = []
#     # colorized = []
#     for encoded_token in sample['source_ids']:
#         code.append(tokenizer.decode(encoded_token, clean_up_tokenization_spaces=False))
#     code[0] = 'START'
#     code = code[:code.index('</s>') + 1]
#     code[-1] = 'END'
#     for attention_layer in sample['attention']:  # number of decoder layers
#         # TODO
#         for i in range(attention_layer.shape[1]):
#             f.write('<br>' + str(i) + '<br>')
#             attention = np.array(attention_layer[0][i].cpu().numpy(), dtype="float64")
#             attention = attention[:len(code)]
#             normalized_attention = ((attention - min(attention)) / (max(attention) - min(attention)))
#             # colorized.append(colorize(code, normalized_attention))
#             colorized = colorize(code, normalized_attention)
#             f.write(colorized)
#             f.write('<br><br><br>')
#         break
#     f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_files", help="path to the pickle files that we want to sample, seperated by comma")
    parser.add_argument("--lang", help="source language", required=True)
    parser.add_argument("--model_name", help="Codebert or GraphCodebert", choices=['Codebert', 'GraphCodebert'],
                        required=True)
    parser.add_argument("--target_indices_csv", help="The csv file containing the target indices for analyze",
                        default=None)
    args = parser.parse_args()

    now = datetime.now()
    print("now =", now)

    model_type = 'roberta'
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    _, _, tokenizer_class = MODEL_CLASSES[model_type]

    if args.model_name == 'Codebert':
        model_name_or_path = 'microsoft/codebert-base'
        tokenizer = tokenizer_class.from_pretrained('roberta-base', do_lower_case=False)
    elif args.model_name == 'GraphCodebert':
        model_name_or_path = 'microsoft/graphcodebert-base'
        tokenizer = tokenizer_class.from_pretrained('microsoft/graphcodebert-base', do_lower_case=False)

    target_indices = []
    if args.target_indices_csv is not None:
        file = open(args.target_indices_csv, mode='r')
        lines = file.readlines()
        for line in lines:
            target_indices.append(int(line.strip()))

    # load parsers
    parsers = {}
    for lang in dfg_function:
        if os.name == 'nt':
            LANGUAGE = Language('parser1/my-languages.so', lang)
        elif os.name == 'posix':
            LANGUAGE = Language('parser1/my-languages.so', lang)
        else:
            print('os problem')
            quit()
        parser = Parser()
        parser.set_language(LANGUAGE)
        parser = [parser, dfg_function[lang]]
        parsers[lang] = parser

    LANGUAGE_KEYWORDS = FIXED_KEYWORDS[args.lang]
    error = 0
    categories_occurrences_total = [
        {"method_name": 0, 'input_variables': 0, 'method_call': 0, 'variable': 0, 'type_identifier': 0,
         'language_keywords': 0,
         'others': 0, 'total': 0} for _ in range(6)]
    categories_score_total = [
        {"method_name": 0.0, 'input_variables': 0.0, 'method_call': 0.0, 'variable': 0.0, 'type_identifier': 0.0,
         'language_keywords': 0.0, 'others': 0.0, 'total': 0.0} for _ in range(6)]
    total_counter = {"method_name": 0, 'input_variables': 0, 'method_call': 0, 'variable': 0, 'type_identifier': 0,
                     'language_keywords': 0,
                     'others': 0, 'total': 0}
    others = Counter()
    files = str(args.pickle_files).split(',')
    language = args.lang

    for pickle_file in files:
        f = open(pickle_file, "rb")
        loaded_obj = pickle.load(f)
        for idx, sample in enumerate(loaded_obj):
            if len(target_indices) is not 0:
                if idx not in target_indices:
                    continue
            try:
                # colorize_in_html(sample)
                code_tokens, code_tokens_tokenized, ori2cur_pos, cur_pos2ori, types, important_tokens = process_example(
                    sample, parsers, language)

                for token in code_tokens:
                    important_type = important_tokens.get(token)
                    if important_type is not None:
                        total_counter[important_type] += 1
                    # elif list(filter(token.startswith, CONTROL_COMMANDS)):
                    elif str(token).strip() in LANGUAGE_KEYWORDS:
                        total_counter['language_keywords'] += 1
                    else:
                        total_counter['others'] += 1
                        others.update([token])

                sum_of_attention_for_tokens_for_all_layers = sum_of_attention_for_tokens(sample, tokenizer.eos_token_id,
                                                                                         ori2cur_pos)
                for sum_of_attention_for_tokens_single_layer, categories_score_for_single_layer in zip(
                        sum_of_attention_for_tokens_for_all_layers, categories_score_total):
                    for token_score in sum_of_attention_for_tokens_single_layer.items():
                        objective_token = code_tokens[token_score[0]]
                        important_type = important_tokens.get(objective_token)
                        if important_type is not None:
                            categories_score_for_single_layer[important_type] += token_score[1]
                        elif str(objective_token).strip() in LANGUAGE_KEYWORDS:
                            categories_score_for_single_layer['language_keywords'] += token_score[1]
                        else:
                            categories_score_for_single_layer['others'] += token_score[1]

                # Based on top-10
                # highest_indices_for_each_layer = indices_for_highest_attentions(sample, tokenizer.eos_token_id,
                #                                                                 ori2cur_pos)
                # for highest_indices_for_this_layer, categories_occurrence in zip(highest_indices_for_each_layer,
                #                                                                  categories_occurrences_total):
                #     high_attention_tokens = []
                #     for index in highest_indices_for_this_layer:
                #         high_attention_tokens.append(code_tokens[index])
                #     for token in high_attention_tokens:
                #         important_type = important_tokens.get(token)
                #         if important_type is not None:
                #             categories_occurrence[important_type] += 1
                #         elif str(token).strip() in LANGUAGE_KEYWORDS:
                #             categories_occurrence['language_keywords'] += 1
                #         else:
                #             categories_occurrence['others'] += 1
                if 'idx' in sample:
                    print(str(sample['idx']) + ' \\ ' + str(len(loaded_obj)))
                    # if sample['idx'] > 5:
                    #     break
                else:
                    print(str(idx) + ' \\ ' + str(len(loaded_obj)))
            except Exception as e:
                print(e)
                error += 1
                pass
        f.close()
        del loaded_obj

    total = 0
    for value in total_counter.values():
        total += value
    total_counter['total'] = total

    # for each_layer in categories_score_total:
    #     total = 0
    #     for value in each_layer.values():
    #         total += value
    #     each_layer['total'] = total
    print(str(error) + ' files missing.')
    # for layer in categories_score_total:
    #     print(layer)
    print('\ntotal tokens distribution:')
    print(total_counter)

    df = pd.DataFrame(categories_score_total)
    df_total = pd.DataFrame([total_counter] * 6)
    df_normalized = df / df_total
    df_normalized ['total'] = df_normalized.sum(axis=1)
    df_normalized_percentage = df_normalized.iloc[:, :].div(df_normalized.total, axis=0)
    df_normalized_percentage = df_normalized_percentage*100


    print(df.to_string())
    print(df_normalized.to_string())
    print(df_normalized_percentage.to_string())
    # Based on top-10
    # for each_layer in categories_occurrences_total:
    #     total = 0
    #     for value in each_layer.values():
    #         total += value
    #     each_layer['total'] = total
    # print(str(error) + ' files missing.')
    # for layer in categories_occurrences_total:
    #     print(layer)
    # print('\ntotal tokens distribution:')
    # print(total_counter)
    #
    # df = pd.DataFrame(categories_occurrences_total)
    # df_perc = df.iloc[:, :].div(df.total, axis=0)
    # df_total = pd.DataFrame([total_counter])


    files = str(args.pickle_files).split('/')
    # writer = pd.ExcelWriter(os.path.join(files[0], "RQ2_results_score_based.xlsx"), engine='xlsxwriter')
    # df.to_excel(writer, sheet_name='layers')
    # df_perc.to_excel(writer, sheet_name='layers_percentage')
    # df_total.to_excel(writer, sheet_name='total count')
    # df.to_excel(writer, sheet_name='total_scores')
    # df_normalized.to_excel(writer, sheet_name='total_scores_normalized')
    # df_normalized_percentage.to_excel(writer, sheet_name='total_scores_normalized_perc')
    # df_total.to_excel(writer, sheet_name='number_of_tokens_per_category')
    # writer.save()

    now = datetime.now()
    print("now =", now)
