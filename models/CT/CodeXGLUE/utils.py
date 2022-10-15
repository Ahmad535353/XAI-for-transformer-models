# import asttokens, ast
import re
import traceback
from collections import Counter

import matplotlib
import torch
from tree_sitter import Language, Parser

from parser1 import tree_to_token_index, index_to_code_token

# a = Language.build_library('my-languages', ['tree-sitter-java-master'])
# JAVA_LANG = Language('my-languages.dll', 'java')

CONTROL_COMMANDS = ['if', 'else', 'switch', 'case', 'while', 'class', 'enum', 'interface', 'byte', 'short', 'char',
                    'int', 'long', 'float', 'double', 'boolean', 'Annotation', 'public', 'protected', 'private',
                    'static', 'abstract', 'final', 'native', 'synchronized', 'transient', 'volatile', 'strictfp',
                    'assert', 'return', 'throw', 'try', 'catch', 'finally', 'default', 'super', 'while', 'do', 'for',
                    'break', 'continue', 'super', 'void', 'byte', 'short', 'char', 'int', 'long', 'float', 'double',
                    'boolean', 'import']


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

def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('Blues')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string


# def tree_to_token_index(root_node):
#     if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
#         return [(root_node.start_point,root_node.end_point)]
#     else:
#         code_tokens=[]
#         for child in root_node.children:
#             code_tokens+=tree_to_token_index(child)
#         return code_tokens

# def map_indices(sample):
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

def indices_for_highest_attentions(sample_with_attentions, eos_token_id):
    highest_indices_for_each_layer = []
    END_INDEX = sample_with_attentions['source_ids'].index(eos_token_id)
    for attention_layer in sample_with_attentions['attention']:
        # indices = set()
        indices = []
        attention_layer = attention_layer[0]
        for output_token_attentions in attention_layer:
            top_20 = torch.topk(output_token_attentions, END_INDEX-1, largest=True)
            top_20 = top_20.indices.tolist()
            top_3 = []
            while len(top_3) < 4:
                temp = top_20.pop(0)
                if temp == 0 or temp >= END_INDEX:
                     continue
                top_3.append(temp)
            indices += top_3
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


def extract_important_tokens(root_node):
    important_tokens = {}
    method_name = str(root_node.children[0].text.decode('utf8')).split()[-1]
    input_variables = str(root_node.children[1].text.decode('utf8'))[1:-1]
    if input_variables is not ' ':
        input_variables = input_variables.split(',')
        input_variables = [x.strip().split(' ')[-1] for x in input_variables]
    else:
        input_variables = []

    important_tokens[method_name] = 'method_name'
    for x in input_variables:
        important_tokens[x] = 'input_variables'

    text = str(root_node.text.decode('utf8'))

    # variables = re.findall(r'VAR_\d+', text)
    # methods = re.findall(r'METHOD_\d+', text)
    # for x in variables:
    #     if x not in important_tokens:
    #         important_tokens[x] = 'variable'
    # for x in methods:
    #     if x not in important_tokens:
    #         important_tokens[x] = 'method_call'

    for node in traverse_tree(root_node.children[-1]):
        if node.text.decode("utf-8") in important_tokens:
            continue
        if node.type == 'identifier':
            if node.next_sibling is None:
                important_tokens[node.text.decode("utf-8")] = 'variable'
                continue
            if node.next_sibling.type == 'argument_list':
                important_tokens[node.text.decode("utf-8") ] = 'method_call'
            else:
                important_tokens[node.text.decode("utf-8") ] = 'variable'
        # if node.type == 'local_variable_declaration':
        #     temp_str = str(node.text.decode('utf8'))
        #     if temp_str.endswith(';'):
        #         temp_str = temp_str[:-1]
        #         if '=' in temp_str:
        #             temp_str = temp_str[:temp_str.index('=')].strip().split()[-1]
        #         else:
        #             temp_str = temp_str.strip().split()[-1]
        #         if temp_str not in important_tokens:
        #             important_tokens[temp_str] = 'local_variable'
        #     else:
        #         continue
        # elif node.type == 'method_invocation':
        #     temp_str = str(node.text.decode('utf8'))
        #     if '(' in temp_str:
        #         temp_str = temp_str[:temp_str.index('(')].strip().split()[-1]
        #         if temp_str not in important_tokens:
        #             important_tokens[temp_str] = 'method_invocation'
    return important_tokens


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
        important_tokens = extract_important_tokens(root_node)
        tokens_index_with_type = tree_to_token_index(root_node)

        tokens_index = [x[0:2] for x in tokens_index_with_type]
        types = [x[2] for x in tokens_index_with_type]
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
