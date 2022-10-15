import asttokens, ast
import json
import pickle

import numpy as np
import unicodedata

# import numpy as np
from utils import pre_process_nl, pre_process_code, get_attentions_of_ast_index


def match_tokens_to_attentions(ast_tree, tokens_decoded):
    ast_tokens = ast_tree._tokens[:-3]
    mapped_indices = {}

    ast_token_index = 0
    tokens_decoded_index = 1
    current_string = tokens_decoded[tokens_decoded_index]
    current_decoded_indices = [1]
    while tokens_decoded[tokens_decoded_index] != 'END':
        try:
            ast_string = ast_tokens[ast_token_index].string
            # if current_string.startswith('\'####POST-PROCESSING"{0}'):
            # print('here')
        except IndexError:
            # print('error here!')
            raise IndexError
        if ast_tokens[ast_token_index].type == 60 or ast_tokens[ast_token_index].type == 3:

            ast_string = ast_string.replace(" ", "")
            ast_string = ast_string.replace("\n", "")
            ast_string = ast_string.replace("\t", "")
            ast_string = ast_string.replace(u'\xa0', "")
            if 'з' in ast_string or 'л' in ast_string:
                break
        if ast_string == current_string:
            # current_decoded_indices.append(tokens_decoded_index)
            mapped_indices[ast_token_index] = current_decoded_indices.copy()
            ast_token_index += 1
            tokens_decoded_index += 1
            current_string = tokens_decoded[tokens_decoded_index].strip()
            current_decoded_indices.clear()
            current_decoded_indices.append(tokens_decoded_index)
        elif ast_string.startswith(current_string):
            # current_decoded_indices.append(tokens_decoded_index)
            tokens_decoded_index += 1
            current_string += tokens_decoded[tokens_decoded_index].strip()
            if current_string == '* *':
                current_string = '**'
            current_decoded_indices.append(tokens_decoded_index)
        else:
            ast_token_index += 1
    if len(current_decoded_indices) > 1:  # because maybe END has been appended since 256 limit was middle of a token
        mapped_indices[ast_token_index] = current_decoded_indices[:-1].copy()
    return mapped_indices


def map_indices(sample):
    try:
        ast_tree = asttokens.ASTTokens(sample['natural_code'], parse=True)
    except SyntaxError as e:
        # bad_samples.append(sample['idx'])
        print(str(sample['idx']) + '\tsyntax-error')
        return None, True
    # attentions = sample['attentions']
    tokens_decoded = sample['source_code_tokens_decoded']
    try:
        mapped_indices = match_tokens_to_attentions(ast_tree, tokens_decoded)
    except IndexError:
        # bad_samples.append(sample['idx'])
        print(str(sample['idx']) + '\tbad_sample')
        return None, True
    ast_tokens = ast_tree._tokens
    absent_tokens = []
    for i in range(len(ast_tokens)):
        ast_t = ast_tokens[i].string
        if i not in mapped_indices:
            absent_tokens.append(ast_t)
            continue
        decoded_t = [tokens_decoded[a].strip() for a in mapped_indices[i]]
        decoded_t = ''.join(decoded_t).strip()
        if ast_t != decoded_t:
            ast_cleaned = ast_t.replace(" ", "").replace("\n", "").replace("\t", "").replace(u'\xa0', "")
            # if ast_cleaned == '#символыужепредпосчитаны':
            #     continue
            if not ast_cleaned.startswith(decoded_t):
                print('\nat index:' + str(sample['idx']))
                print('ast says:' + ast_t)
                print('but decoded says:' + decoded_t + '\n')
                # raise AssertionError
    return mapped_indices, False
    # print('absent tokens:')
    # print(' '.join(absent_tokens).strip())


def extract_important_tokens(ast_tree):
    function_names = set()
    args = set()
    # return_values = set()
    # control_flow_tokens = set()
    function_calls = set()
    variable_names = set()
    # nodes = []
    for node in asttokens.util.walk(ast_tree.tree):
        # nodes.append(node)
        if type(node).__name__ == 'FunctionDef':
            function_names.add(node.name)
        elif type(node).__name__ == 'Name':
            variable_names.add(node.id)
        elif type(node).__name__ == 'arg':
            args.add(node.arg)
        elif type(node).__name__ == 'Call':
            if type(node.func).__name__ == 'Name':
                function_calls.add(node.func.id)
            elif type(node.func).__name__ == 'Attribute':
                function_calls.add(node.func.attr)
        # elif type(node).__name__ == 'Return':
        #     if node.value is not None:
        #         return_values.add(ast_tree.get_text(node).replace('return', '').strip())
    variable_names.discard(x for x in function_names)
    variable_names.discard(x for x in args)
    for x in function_calls:
        variable_names.discard(x)
    important_tokens = {"function_names": function_names, 'args': args, 'function_calls': function_calls,
                        'variable_names': variable_names}
    return important_tokens


def category_contribution(important_tokens, highest_attention_tokens_in_ast):
    categories_occurrence = {"function_names": 0, 'args': 0, 'function_calls': 0, 'variable_names': 0, 'comment': 0,
                             'control_commands': 0, 'total': len(highest_attention_tokens_in_ast)}

    for high_attention_token in highest_attention_tokens_in_ast:
        high_attention_token = high_attention_token.strip()
        if high_attention_token in important_tokens['function_names']:
            categories_occurrence["function_names"] += 1
        elif high_attention_token in important_tokens['args']:
            categories_occurrence["args"] += 1
        elif high_attention_token in important_tokens['function_calls']:
            categories_occurrence["function_calls"] += 1
        elif high_attention_token in important_tokens['variable_names']:
            categories_occurrence["variable_names"] += 1
        elif high_attention_token.startswith('#'):
            categories_occurrence["comment"] += 1
        elif high_attention_token.startswith(
                ('if', 'for', 'while', 'break', 'continue', 'try', 'except', 'with', 'return')):
            categories_occurrence["control_commands"] += 1

    return categories_occurrence


def analyze_sample(sample_with_attentions, mapped_indices_for_sample1, parsed_code_for_sample):
    important_tokens = extract_important_tokens(parsed_code_for_sample)

    indices = set()
    for attention_layer in sample_with_attentions['attentions']:
        highest_values = sorted(attention_layer, reverse=True)[:5]
        highest_indices = {attention_layer.index(x) for x in highest_values}
        indices |= highest_indices
    END_index = sample_with_attentions['source_code_tokens_decoded'].index('END')
    indices.discard(0)
    indices.discard(END_index)

    highest_attention_tokens = [sample_with_attentions['source_code_tokens_decoded'][x] for x in indices]
    try:
        highest_indices_in_ast = [mapped_indices_for_sample1['mapped_from_decoder_to_ast'][x] for x in indices]
    except KeyError as e:
        print(e)
        return {"function_names": 0, 'args': 0, 'function_calls': 0, 'variable_names': 0, 'comment': 0,
                'control_commands': 0, 'total': 00}
    highest_attention_tokens_in_ast = [parsed_code_for_sample.tokens[x].string for x in highest_indices_in_ast]
    lines = [parsed_code_for_sample.tokens[x].line for x in highest_indices_in_ast]

    categories_occurrences = category_contribution(important_tokens, highest_attention_tokens_in_ast)
    return categories_occurrences


pickle_file = 'model/python/prediction_pickle.pickle'
mapped_indices_and_bad_samples_pickle = 'model/python/mapped_indices_and_bad_samples_pickle.pickle'


def find_predicted_words_indices_in_ast_code(source_code, best_prediction):
    prediction_index_to_ast_source_index = dict()
    ast_source_indices = set()
    temp = dict()
    for idx1, token in enumerate(best_prediction):
        clean_token = pre_process_nl(token)
        if clean_token is 0:
            continue
        for idx2, code_toekn in enumerate(source_code):
            clean_code_token = pre_process_code(code_toekn)
            if clean_code_token is 0:
                continue
            if clean_token in clean_code_token:
                if idx1 in prediction_index_to_ast_source_index:
                    ast_source_indices.add(idx2)
                    prediction_index_to_ast_source_index[idx1].append(idx2)
                    temp[token].append(code_toekn)
                else:
                    ast_source_indices.add(idx2)
                    prediction_index_to_ast_source_index[idx1] = [idx2]
                    temp[token] = [code_toekn]
    return ast_source_indices, prediction_index_to_ast_source_index, temp


if __name__ == '__main__':
    phase = 3
    # phase 1
    if phase == 1:
        f = open(pickle_file, "rb")
        loaded_obj = pickle.load(f)
        mapped_indices = []
        bad_samples = []
        for sample in loaded_obj:
            mapped_indices_ast_to_decoded, bad_sample_flag = map_indices(sample)
            inv_map_decoded_to_ast = dict()
            if mapped_indices_ast_to_decoded is None:
                inv_map_decoded_to_ast = None
            else:
                for key, values in mapped_indices_ast_to_decoded.items():
                    for value in values:
                        inv_map_decoded_to_ast[value] = key
            mapped_indices_for_sample = {'index': sample['idx'],
                                         'mapped_from_ast_to_decoder': mapped_indices_ast_to_decoded,
                                         'mapped_from_decoder_to_ast': inv_map_decoded_to_ast}
            mapped_indices.append(mapped_indices_for_sample)
            if bad_sample_flag:
                bad_samples.append(sample['idx'])
        with open(mapped_indices_and_bad_samples_pickle, 'wb') as mapped_indices_and_bad_samples_pickle_file:
            pickle_entry = {'mapped_indices': mapped_indices, 'bad_samples': bad_samples}
            pickle.dump(pickle_entry, mapped_indices_and_bad_samples_pickle_file)
    elif phase == 2:
        file = open(mapped_indices_and_bad_samples_pickle, "rb")
        mapped_indices_obj = pickle.load(file)
        mapped_indices = mapped_indices_obj['mapped_indices']
        bad_samples = mapped_indices_obj['bad_samples']

        # problematic samples
        bad_samples.append(273)

        f = open(pickle_file, "rb")
        loaded_obj = pickle.load(f)

        categories_occurrences_total = {"function_names": 0, 'args': 0, 'function_calls': 0, 'variable_names': 0,
                                        'comment': 0,
                                        'control_commands': 0, 'total': 0}
        for sample in loaded_obj:
            sample_index = sample['idx']
            if sample_index in bad_samples:
                continue
            # TODO: natural_code or source_code?
            parsed_code = asttokens.ASTTokens(sample['natural_code'], parse=True)
            categories_occurrences = analyze_sample(sample, mapped_indices[sample_index], parsed_code)
            for key, value in categories_occurrences.items():
                categories_occurrences_total[key] += value
            print(categories_occurrences_total)

    elif phase == 3:
        file = open(mapped_indices_and_bad_samples_pickle, "rb")
        mapped_indices_obj = pickle.load(file)
        mapped_indices = mapped_indices_obj['mapped_indices']
        bad_samples = mapped_indices_obj['bad_samples']
        # problematic samples
        bad_samples.append(273)
        f = open(pickle_file, "rb")
        loaded_obj = pickle.load(f)
        counter = np.zeros([6, 4])
        for sample in loaded_obj:
            sample_index = sample['idx']
            if sample_index%1000 == 0:
                print(sample_index)
            if sample_index in bad_samples:
                continue
            parsed_code = asttokens.ASTTokens(sample['natural_code'], parse=True)
            ast_code_tokens = [x.string for x in parsed_code.tokens]
            prediction_tokens = sample['best_prediction'].split(' ')
            ast_source_indices_appear_in_prediction, prediction_index_to_ast_source_index, human_readable = find_predicted_words_indices_in_ast_code(
                ast_code_tokens, prediction_tokens)
            ast_indices_for_methods_names = get_ast_indices_for_methods_names()

            attentions = np.array(sample['attentions'])
            attentions[attentions == 0] = np.nan
            first_quadrant_tresholds = np.nanquantile(attentions, 0.1, axis=1, keepdims=True)
            second_quadrant_tresholds = np.nanquantile(attentions, 0.5, axis=1, keepdims=True)
            third_quadrant_tresholds = np.nanquantile(attentions, 0.9, axis=1, keepdims=True)
            quadrant_tresholds = np.concatenate((third_quadrant_tresholds, second_quadrant_tresholds, first_quadrant_tresholds), axis=1)
            for ast_index in ast_source_indices_appear_in_prediction:
                if ast_index not in mapped_indices[sample_index]['mapped_from_ast_to_decoder']:
                    continue
                attention_for_ast_index = get_attentions_of_ast_index(ast_index, mapped_indices[sample_index][
                    'mapped_from_ast_to_decoder'], sample['attentions'])
                # if any x in attention_for_ast_index
                # comparison = np.array(attention_for_ast_index).reshape(6, 1) > third_quadrant_tresholds
                np_attentions = np.array(attention_for_ast_index).reshape(6, 1)
                for attention_layer, quadrant_treshold, counter_idx in zip(np_attentions, quadrant_tresholds, range(len(np_attentions))):
                    if attention_layer > quadrant_treshold[0]:
                        counter[counter_idx][0] += 1
                    elif attention_layer > quadrant_treshold[1]:
                        counter[counter_idx][1] += 1
                    elif attention_layer > quadrant_treshold[2]:
                        counter[counter_idx][2] += 1
                    else:
                        counter[counter_idx][3] += 1
        print(counter)
