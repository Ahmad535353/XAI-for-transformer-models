# import asttokens, ast
import json
import keyword
import re
import string
import traceback
from collections import Counter

# import matplotlib
import numpy as np
import pandas as pd
import torch
from nltk import WordNetLemmatizer
from tree_sitter import Language, Parser

from parser1 import tree_to_token_index, index_to_code_token

# a = Language.build_library('my-languages', ['tree-sitter-java-master'])
# JAVA_LANG = Language('my-languages.dll', 'java')

FIXED_KEYWORDS = {
    'java': ['if', 'else', 'switch', 'case', 'while', 'class', 'enum', 'interface', 'Annotation', 'public',
             'protected', 'private', 'static', 'abstract', 'final', 'native', 'synchronized', 'transient',
             'volatile', 'strictfp', 'assert', 'return', 'throw', 'try', 'catch', 'finally', 'default', 'super',
             'do', 'for', 'break', 'continue', 'super', 'void', 'import', 'extends', 'implements', 'import',
             'instanceof', 'new', 'null', 'package', 'this', 'throws'], 'python': keyword.kwlist}


# JAVA_TYPES = ['byte', 'short', 'char', 'int', 'long', 'float', 'double', 'boolean']


# CONTROL_COMMANDS = ['if', 'else', 'switch', 'case', 'while', 'class', 'enum', 'interface', 'Annotation', 'public',
#                     'protected', 'private', 'static', 'abstract', 'final', 'native', 'synchronized', 'transient',
#                     'volatile', 'strictfp', 'assert', 'return', 'throw', 'try', 'catch', 'finally', 'default', 'super',
#                     'while', 'do', 'for', 'break', 'continue', 'super', 'void', 'import']
# PYTHON_KEYWORDS = [keyword.kwlist]

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 natural_code=None
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.natural_code = natural_code

#
# def get_tokens(node):
#     tokens = []
#     if node.child_count == 1:
#         return node.text.decode()
#     for child in node.children:
#         temp = get_tokens(child)
#         if isinstance(temp, str):
#             tokens.append(temp)
#         else:
#             tokens.extend(temp)
#         # for temp_token in temp_tokens:
#         #     tokens.append(temp_token)
#     return tokens

# def colorize(words, color_array):
#     # words is a list of words
#     # color_array is an array of numbers between 0 and 1 of length equal to words
#     cmap = matplotlib.cm.get_cmap('Blues')
#     template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
#     colored_string = ''
#     for word, color in zip(words, color_array):
#         color = matplotlib.colors.rgb2hex(cmap(color)[:3])
#         colored_string += template.format(color, '&nbsp' + word + '&nbsp')
#     return colored_string


# def tree_to_token_index(root_node):
#     if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
#         return [(root_node.start_point,root_node.end_point)]
#     else:
#         code_tokens=[]
#         for child in root_node.children:
#             code_tokens+=tree_to_token_index(child)
#         return code_tokens

# def map_indices(1mple):
#     parser = Parser()
#     parser.set_language(JAVA_LANG)
#     try:
#         ast_tree = parser.parse(bytes(sample['source'], "utf8"))
#         # ast_tree = javalang.parse.parse(sample['source'])
#     except SyntaxError as e:
#         # bad_samples.append(sample['idx'])
#         print(str(sample['idx']) + '\tsyntax-error')
#         return None, True
#     root_node = ast_tree.root_node
#     tokens_index = tree_to_token_index(root_node)
#
#     tokens_decoded = sample['source_code_tokens_decoded']
#     try:
#         mapped_indices = match_tokens_to_attentions(ast_tree, tokens_decoded)
#     except IndexError:
#         # bad_samples.append(sample['idx'])
#         print(str(sample['idx']) + '\tbad_sample')
#         return None, True
#     ast_tokens = ast_tree._tokens
#     absent_tokens = []
#     for i in range(len(ast_tokens)):
#         ast_t = ast_tokens[i].string
#         if i not in mapped_indices:
#             absent_tokens.append(ast_t)
#             continue
#         decoded_t = [tokens_decoded[a].strip() for a in mapped_indices[i]]
#         decoded_t = ''.join(decoded_t).strip()
#         if ast_t != decoded_t:
#             ast_cleaned = ast_t.replace(" ", "").replace("\n", "").replace("\t", "").replace(u'\xa0', "")
#             # if ast_cleaned == '#символыужепредпосчитаны':
#             #     continue
#             if not ast_cleaned.startswith(decoded_t):
#                 print('\nat index:' + str(sample['idx']))
#                 print('ast says:' + ast_t)
#                 print('but decoded says:' + decoded_t + '\n')
#                 # raise AssertionError
#     return mapped_indices, False
#     # print('absent tokens:')
#     # print(' '.join(absent_tokens).strip())

# def match_tokens_to_attentions(ast_tree, tokens_decoded):
#     ast_tokens = ast_tree._tokens[:-3]
#     mapped_indices = {}
#
#     ast_token_index = 0
#     tokens_decoded_index = 1
#     current_string = tokens_decoded[tokens_decoded_index]
#     current_decoded_indices = [1]
#     while tokens_decoded[tokens_decoded_index] != 'END':
#         try:
#             ast_string = ast_tokens[ast_token_index].string
#             # if current_string.startswith('\'####POST-PROCESSING"{0}'):
#             # print('here')
#         except IndexError:
#             # print('error here!')
#             raise IndexError
#         if ast_tokens[ast_token_index].type == 60 or ast_tokens[ast_token_index].type == 3:
#
#             ast_string = ast_string.replace(" ", "")
#             ast_string = ast_string.replace("\n", "")
#             ast_string = ast_string.replace("\t", "")
#             ast_string = ast_string.replace(u'\xa0', "")
#             if 'з' in ast_string or 'л' in ast_string:
#                 break
#         if ast_string == current_string:
#             # current_decoded_indices.append(tokens_decoded_index)
#             mapped_indices[ast_token_index] = current_decoded_indices.copy()
#             ast_token_index += 1
#             tokens_decoded_index += 1
#             current_string = tokens_decoded[tokens_decoded_index].strip()
#             current_decoded_indices.clear()
#             current_decoded_indices.append(tokens_decoded_index)
#         elif ast_string.startswith(current_string):
#             # current_decoded_indices.append(tokens_decoded_index)
#             tokens_decoded_index += 1
#             current_string += tokens_decoded[tokens_decoded_index].strip()
#             if current_string == '* *':
#                 current_string = '**'
#             current_decoded_indices.append(tokens_decoded_index)
#         else:
#             ast_token_index += 1
#     if len(current_decoded_indices) > 1:  # because maybe END has been appended since 256 limit was middle of a token
#         mapped_indices[ast_token_index] = current_decoded_indices[:-1].copy()
#     return mapped_indices


# def index_to_code_token(index, code):
#     start_point = index[0]
#     end_point = index[1]
#     if start_point[0] == end_point[0]:
#         s = code[start_point[0]][start_point[1]:end_point[1]]
#     else:
#         s = ""
#         s += code[start_point[0]][start_point[1]:]
#         for i in range(start_point[0] + 1, end_point[0]):
#             s += code[i]
#         s += code[end_point[0]][:end_point[1]]
#     return s
# def top_k(x, k):
#     ind = np.argpartition(x, -1 * k)[-1 * k:]
#     return ind[np.argsort(x[ind])]

def sum_of_attention_for_tokens(sample_with_attentions, eos_token_id, ori2cur_pos_dict):
    highest_indices_for_each_layer = []
    END_INDEX = sample_with_attentions['source_ids'].index(eos_token_id)
    sum_of_attention_for_token_for_each_layer = []
    for attention_layer in sample_with_attentions['attention']:
        # indices = set()
        indices = []
        # attention_layer = attention_layer[0].cpu().numpy()
        px = pd.DataFrame(attention_layer[0].cpu().numpy())
        a = pd.DataFrame()

        # START token has index -1 in map so it will be ignored
        # END token does not have a key in ori2cur_pos_dict so it will be ignored
        for map in ori2cur_pos_dict:
            if map == -1:
                continue
            # b = px.loc[:, ori2cur_pos_dict[map][0]+1:ori2cur_pos_dict[map][1]].mean(axis=1)
            b = px.loc[:, ori2cur_pos_dict[map][0] + 1:ori2cur_pos_dict[map][1]].sum(axis=1)
            a[map] = b
        # a will not have the attentions for START and END token as we want

        sum_of_attention_for_token = a.sum()
        sum_of_attention_for_token_for_each_layer.append(sum_of_attention_for_token)
    return sum_of_attention_for_token_for_each_layer

    #     for index, output_token_attentions in a.iterrows():
    #         # top_k = output_token_attentions.nlargest(5)
    #         top_k = output_token_attentions.sort_values(ascending=False)
    #         # top_20 = torch.topk(output_token_attentions, END_INDEX-1, largest=True)
    #         # top_20 = top_20.indices.tolist()
    #         # top_3 = []
    #         # while len(top_3) < 4:
    #         #     temp = top_20.pop(0)
    #         #     if temp == 0 or temp >= END_INDEX:
    #         #          continue
    #         #     top_3.append(temp)
    #         indices += top_k[:10].index.values.tolist()
    #     counter = Counter(indices)
    #     top_k_total_for_this_layer = counter.most_common(10)
    #     top_k_total_for_this_layer = [x[0] for x in top_k_total_for_this_layer]
    #     highest_indices_for_each_layer.append(top_k_total_for_this_layer)
    # return highest_indices_for_each_layer


def indices_for_highest_attentions(sample_with_attentions, eos_token_id, ori2cur_pos_dict):
    highest_indices_for_each_layer = []
    END_INDEX = sample_with_attentions['source_ids'].index(eos_token_id)
    for attention_layer in sample_with_attentions['attention']:
        # indices = set()
        indices = []
        # attention_layer = attention_layer[0].cpu().numpy()
        px = pd.DataFrame(attention_layer[0].cpu().numpy())
        a = pd.DataFrame()

        # START token has index -1 in map so it will be ignored
        # END token does not have a key in ori2cur_pos_dict so it will be ignored
        for map in ori2cur_pos_dict:
            if map == -1:
                continue
            # b = px.loc[:, ori2cur_pos_dict[map][0]+1:ori2cur_pos_dict[map][1]].mean(axis=1)
            b = px.loc[:, ori2cur_pos_dict[map][0] + 1:ori2cur_pos_dict[map][1]].sum(axis=1)
            a[map] = b
        # a will not have the attentions for START and END token as we want

        for index, output_token_attentions in a.iterrows():
            # top_k = output_token_attentions.nlargest(5)
            top_k = output_token_attentions.sort_values(ascending=False)
            # top_20 = torch.topk(output_token_attentions, END_INDEX-1, largest=True)
            # top_20 = top_20.indices.tolist()
            # top_3 = []
            # while len(top_3) < 4:
            #     temp = top_20.pop(0)
            #     if temp == 0 or temp >= END_INDEX:
            #          continue
            #     top_3.append(temp)
            indices += top_k[:10].index.values.tolist()
        counter = Counter(indices)
        top_k_total_for_this_layer = counter.most_common(10)
        top_k_total_for_this_layer = [x[0] for x in top_k_total_for_this_layer]
        highest_indices_for_each_layer.append(top_k_total_for_this_layer)
    return highest_indices_for_each_layer


def traverse_tree(tree):
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False


def traverse_tree_test(tree):
    test = 0
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        yield cursor.node, test

        if cursor.goto_first_child():
            test += 1
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            else:
                test -= 1

            if cursor.goto_next_sibling():
                retracing = False


def get_all_identifiers(root_node):
    cursor = root_node.walk()
    reached_root = False
    while reached_root == False:
        yield cursor.node
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False


def extract_important_tokens(root_node, lang):
    important_tokens = {}
    comments = {}
    if lang == 'python':
        idents = []
        for node in traverse_tree(root_node.children[0]):
            if node.type == 'identifier':
                idents.append(node.text.decode("utf-8"))
            elif node.type == ':':
                break
        method_name = idents[0]
        idents.pop(0)
        input_variables = idents
    elif lang == 'java':
        method_name = str(root_node.children[0].text.decode('utf8')).split()[-1]
        input_variables = str(root_node.children[1].text.decode('utf8'))[1:-1]
        if input_variables is not ' ':
            input_variables = input_variables.split(',')
            input_variables = [x.strip().split(' ')[-1] for x in input_variables]
        else:
            input_variables = []
    else:
        method_name = []
        input_variables = []
        raise ValueError

    important_tokens[method_name] = 'method_name'
    for x in input_variables:
        important_tokens[x] = 'input_variables'

    for node in traverse_tree(root_node):
        # if node.text.decode("utf-8") == 'boolean':
        #     print('here')
        if node.text.decode("utf-8") in important_tokens:
            continue
        if node.type == 'identifier':
            if node.next_sibling is None:
                important_tokens[node.text.decode("utf-8")] = 'variable'
                continue
            if node.next_sibling.type == 'argument_list':
                important_tokens[node.text.decode("utf-8")] = 'method_call'
            else:
                important_tokens[node.text.decode("utf-8")] = 'variable'
        elif node.type == 'type_identifier' or str(node.type).endswith('_type'):
            important_tokens[node.text.decode("utf-8")] = 'type_identifier'
        elif node.type == 'block_comment' or node.type == 'line_comment':
            comments[node.text.decode("utf-8")] = ['comment', node.start_point, node.end_point]
    return important_tokens, comments


def extract_dataflow(code, parsers, lang):
    # remove comments
    # try:
    #     code=remove_comments_and_docstrings(code,lang)
    # except:
    #     pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parsers[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        important_tokens, comments = extract_important_tokens(root_node, lang)
        if comments:
            print(comments)

        tokens_index_with_type = tree_to_token_index(root_node)

        tokens_index = [x[0:2] for x in tokens_index_with_type]
        types = [None for x in tokens_index_with_type]
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parsers[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except Exception as err:
        print(traceback.format_exc())
        dfg = []
    return code_tokens, dfg, types, important_tokens


def extract_stats(code, parsers, lang):
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parsers[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node

        max_depth = 0
        for node, depth in traverse_tree_test(root_node):
            max_depth = max(max_depth, depth)

        important_tokens, comments = extract_important_tokens(root_node, lang)
        tokens_index_with_type = tree_to_token_index(root_node)

        tokens_index = [x[0:2] for x in tokens_index_with_type]
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    except Exception as err:
        print(traceback.format_exc())
        code_tokens = []
        important_tokens = {}
        max_depth = 0
    return code_tokens, important_tokens, max_depth


def process_example_for_attention_impact(example, parsers, lang, tokenizer):
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

def rank_of_the_similar_token_in_input(sample_with_attentions, tokenizer):
    list_of_ranks_for_all_layers = []

    not_found = 0
    is_first_layer = 1
    for attention_layer in sample_with_attentions['attention']:
        px = pd.DataFrame(attention_layer[0].cpu().numpy())
        input_length = len(sample_with_attentions['source_ids'])
        # list_of_ranks_for_this_layer = Counter()
        list_of_ranks_for_this_layer = []
        for output_token, [index, output_token_attentions] in zip(sample_with_attentions['encoded_output'],
                                                                  px.iterrows()):
            output_token_attentions_sorted = output_token_attentions.sort_values(ascending=False)
            has_found = False
            # print('here1')
            for rank, [attention_index, _] in enumerate(output_token_attentions_sorted.items()):
                # print('here2')
                respective_input_token = sample_with_attentions['source_ids'][attention_index]
                # print('here3')
                # print(output_token)
                # print(int(output_token))
                if tokenizer.decode(respective_input_token).casefold() == tokenizer.decode(int(output_token)).casefold():
                    # print('here4')
                    has_found = True
                    # if rank > 10:
                    #     list_of_ranks_for_this_layer.update([11])
                    # else:
                    normalized_rank = (rank+1) * 100/input_length
                    # print('here4-1')
                    # list_of_ranks_for_this_layer.update([normalized_rank])
                    list_of_ranks_for_this_layer.append(normalized_rank)
                    # print('here4-2')
                    break
                # print('here5')
            if is_first_layer and not has_found:
                # print('here6')
                # list_of_ranks_for_this_layer.update([10000])
                not_found += 1
            # print('here7')
        list_of_ranks_for_all_layers.append(list_of_ranks_for_this_layer)
        # print('here8')
        is_first_layer = 0
    return list_of_ranks_for_all_layers, not_found


def read_examples_from_raw_dataset(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    natural_code=js['code']
                )
            )
    return examples

def pre_process(code_tokens_inp, target_tokens_inp):
    lemmatizer = WordNetLemmatizer()

    code_tokens_cleaned = []
    for token in code_tokens_inp:
        if len(token) < 2:
            continue
        temp = [lemmatizer.lemmatize(w.lower()) for w in token.split('_')]
        code_tokens_cleaned += temp

    punctuations = [char for char in string.punctuation]
    target_tokens_cleaned = set()
    for token in target_tokens_inp:
        if token in punctuations:
            continue
        target_tokens_cleaned.add(lemmatizer.lemmatize(token.lower()))

    return code_tokens_cleaned, target_tokens_cleaned

def calculate_the_overlap(code_tokens_input, target_tokens_input):
    code_tokens_cleaned, target_tokens_cleaned = pre_process(code_tokens_input, target_tokens_input)
    counter = []
    for token in target_tokens_cleaned:
        # if any(token in x for x in code_tokens_sample):
        # changed it
        if any(token in x for x in code_tokens_cleaned):
            counter.append(token)
    overlap = len(counter) / len(target_tokens_cleaned)
    return overlap
