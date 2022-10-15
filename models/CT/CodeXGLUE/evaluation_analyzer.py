import collections
import csv
import difflib
import pickle
import traceback

import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from utils import extract_dataflow, colorize, indices_for_highest_attentions, CONTROL_COMMANDS
from tree_sitter import Language, Parser
from parser1 import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser1 import (remove_comments_and_docstrings,
                     tree_to_token_index,
                     index_to_code_token,
                     tree_to_variable_index)
from collections import Counter
import pandas as pd


files = ['prediction_pickle.pickle']
# files = ['prediction_pickle' + str(x) + '.pickle' for x in range(14)]


def process_example(example, parsers):
    ##extract data flow
    code_tokens, dfg, types, important_tokens = extract_dataflow(example['source'], parsers['java'], 'java')
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


dfg_function = {
    # 'python':DFG_python,
    'java': DFG_java,
    # 'ruby':DFG_ruby,
    # 'go':DFG_go,
    # 'php':DFG_php,
    # 'javascript':DFG_javascript
}


# a = Language.build_library('my-languages',['tree-sitter-java-master'])
# JAVA_LANG = Language('my-languages.dll', 'java')

def colorize_in_html(sample):
    f = open('colorize/' + str(sample['idx']) + '.html', 'w', encoding='utf8')
    f.write('source:')
    f.write('<br>')
    f.write(sample['source'])
    f.write('<br>')
    f.write('target:')
    f.write('<br>')
    f.write(sample['target'])
    f.write('<br>')
    f.write('best prediction:')
    f.write('<br>')
    f.write(sample['prediction'])
    f.write('<br>')
    f.write('<br><br><br>')

    code = []
    # colorized = []
    for encoded_token in sample['source_ids']:
        code.append(tokenizer.decode(encoded_token, clean_up_tokenization_spaces=False))
    code[0] = 'START'
    code = code[:code.index('</s>') + 1]
    code[-1] = 'END'
    for attention_layer in sample['attention']:  # number of decoder layers
        # TODO
        for i in range(attention_layer.shape[1]):
            f.write('<br>' + str(i) + '<br>')
            attention = np.array(attention_layer[0][i].cpu().numpy(), dtype="float64")
            attention = attention[:len(code)]
            normalized_attention = ((attention - min(attention)) / (max(attention) - min(attention)))
            # colorized.append(colorize(code, normalized_attention))
            colorized = colorize(code, normalized_attention)
            f.write(colorized)
            f.write('<br><br><br>')
        break
    f.close()


if __name__ == '__main__':
    model_type = 'roberta'
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    _, _, tokenizer_class = MODEL_CLASSES[model_type]
    model_name_or_path = 'microsoft/codebert-base'
    tokenizer = tokenizer_class.from_pretrained('roberta-base', do_lower_case=False)

    # load parsers
    parsers = {}
    for lang in dfg_function:
        LANGUAGE = Language('parser1/my-languages.dll', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parser = [parser, dfg_function[lang]]
        parsers[lang] = parser

    categories_occurrences_total = [
        {"function_names": 0, 'args': 0, 'function_calls': 0, 'variable_names': 0, 'control_commands': 0, 'others': 0,
         'total': 0} for _ in range(6)]
    total_counter = {"function_names": 0, 'args': 0, 'function_calls': 0, 'variable_names': 0, 'control_commands': 0, 'others': 0,
         'total': 0}
    error = 0
    for file_num, pickle_file in enumerate(files):
        f = open('saved_model_correct_from_cedar/' + pickle_file, "rb")
        loaded_obj = pickle.load(f)
        for idx, sample in enumerate(loaded_obj):
            try:
                # colorize_in_html(sample)
                code_tokens, code_tokens_tokenized, ori2cur_pos, cur_pos2ori, types, important_tokens = process_example(
                    sample, parsers)
                # for token in code_tokens:
                #     total_counter['total'] += 1
                #     important_type = important_tokens.get(token)
                #     if important_type is not None:
                #         if important_type == 'method_name':
                #             total_counter['function_names'] += 1
                #         elif important_type == 'input_variables':
                #             total_counter['args'] += 1
                #         elif important_type == 'method_call':
                #             total_counter['function_calls'] += 1
                #         elif important_type == 'variable':
                #             total_counter['variable_names'] += 1
                #     elif list(filter(token.startswith, CONTROL_COMMANDS)):
                #         total_counter['control_commands'] += 1
                #     else:
                #         total_counter['others'] += 1
                highest_indices_for_each_layer = indices_for_highest_attentions(sample, tokenizer.eos_token_id)
                for highest_indices_for_this_layer, categories_occurrence in zip(highest_indices_for_each_layer,
                                                                                 categories_occurrences_total):
                    high_attention_tokens = []
                    for index in highest_indices_for_this_layer:
                        # -1 is because the indices in attention are shifted by one, because of the START token
                        high_attention_tokens.append(code_tokens[cur_pos2ori[index - 1]])
                    for token in high_attention_tokens:
                        categories_occurrence['total'] += 1
                        important_type = important_tokens.get(token)
                        if important_type is not None:
                            if important_type == 'method_name':
                                categories_occurrence['function_names'] += 1
                            elif important_type == 'input_variables':
                                categories_occurrence['args'] += 1
                            elif important_type == 'method_call':
                                categories_occurrence['function_calls'] += 1
                            elif important_type == 'variable':
                                categories_occurrence['variable_names'] += 1
                        elif list(filter(token.startswith, CONTROL_COMMANDS)):
                            categories_occurrence['control_commands'] += 1
                        else:
                            categories_occurrence['others'] += 1
                print(str(file_num) + ': ' + str(idx))
            except Exception as err:
                print(traceback.format_exc())
                error += 1
                pass
        f.close()
        del loaded_obj


    print(total_counter)

    print(categories_occurrences_total)
    print(str(error) + ' files missing.')

    df = pd.DataFrame(categories_occurrences_total)
    df_perc = df.iloc[:,:].div(df.total, axis=0) * 100

    writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    df_perc.to_excel(writer, sheet_name='Sheet2')
    writer.save()