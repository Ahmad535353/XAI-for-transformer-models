import argparse
import os
import pickle
import statistics
from collections import Counter
from datetime import datetime
import nltk
from tree_sitter import Language, Parser

from utils import extract_dataflow, extract_stats, read_examples_from_raw_dataset, calculate_the_overlap
from parser1 import DFG_python, DFG_java

import lizard
import pylev as pylev
from matplotlib import pyplot as plt

from bleu import compute_bleu

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    # 'ruby':DFG_ruby,
    # 'go':DFG_go,
    # 'php':DFG_php,
    # 'javascript':DFG_javascript
}


def load_parsers():
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
    return parsers


# def get_code_tokens(source_code, parsers):
#     tree = parsers[0].parse(bytes(source_code, 'utf8'))
#     root_node = tree.root_node
#     tokens_index_with_type = tree_to_token_index(root_node)
#     tokens_index = [x[0:2] for x in tokens_index_with_type]
#     code = source_code.split('\n')
#     code_tokens = [index_to_code_token(x, code) for x in tokens_index]
#     return code_tokens


if __name__ == '__main__':
    nltk.download('omw-1.4')
    nltk.download('wordnet')

    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_files", help="path to the pickle files that we want to sample, seperated by comma")
    parser.add_argument("--lang", help="source language", required=True)
    parser.add_argument("--model_name", help="Codebert or GraphCodebert", choices=['Codebert', 'GraphCodebert'],
                        required=True)
    parser.add_argument("--raw_dataset", help="raw dataset for code document generation", required=False, default=None)
    args = parser.parse_args()

    # Reading the raw dataset
    file = args.raw_dataset
    eval_examples = read_examples_from_raw_dataset(file)
    examples_counter = 0

    language = args.lang
    files = str(args.pickle_files).split(',')

    print("now =", datetime.now())

    parsers = load_parsers()

    stats_analyze = {'index': [], 'distances': [], 'CDG_overlap': [], 'bleus': [], 'n_tokens': [], 'cyclomatic_complexity': [],
                     'nested_block_depth': [],
                     'input_variables': [], 'method_call': [], 'variable': []}
    distances = []
    bleus = []

    error = 0

    for pickle_file in files:
        f = open(pickle_file, "rb")
        loaded_obj = pickle.load(f)
        for idx, sample in enumerate(loaded_obj):
            try:
                source_code_tokens = str(sample['source']).strip().split()
                target_code_tokens = str(sample['target']).strip().split()
                prediction_code_tokens = str(sample['prediction']).strip().split()

                if args.raw_dataset is None:
                    source_code = sample['source']
                else:
                    try:
                        assert sample['target'] == eval_examples[examples_counter].target
                    except AssertionError as e:
                        print(e)
                        exit()
                    source_code = eval_examples[examples_counter].natural_code
                    examples_counter += 1
                if language == 'python':
                    i = lizard.analyze_file.analyze_source_code("test.py", source_code)
                elif language == 'java':
                    i = lizard.analyze_file.analyze_source_code("test.java", source_code)
                else:
                    raise Exception("Language is not supported")
                stats = i.function_list[0].__dict__

                code_tokens, important_tokens, max_depth = extract_stats(sample['source'], parsers[language], language)

                counter = Counter(important_tokens.values())

                lev_distance = pylev.levenshtein(source_code_tokens, target_code_tokens)
                # bb = _bleu('saved_model/test_0.gold',
                #            'saved_model/test_0.gold')
                bleu_score, _1, _2, _3, _4, _5 = compute_bleu([[target_code_tokens]], [prediction_code_tokens],
                                                              max_order=4,
                                                              smooth=True)
                CDG_overlap = calculate_the_overlap(code_tokens[:], target_code_tokens[:])

                stats_analyze['index'].append(idx)
                stats_analyze['distances'].append(lev_distance)
                stats_analyze['CDG_overlap'].append(CDG_overlap)
                stats_analyze['bleus'].append(bleu_score)
                stats_analyze['n_tokens'].append(len(code_tokens))
                stats_analyze['cyclomatic_complexity'].append(stats['cyclomatic_complexity'])
                stats_analyze['nested_block_depth'].append(max_depth)
                stats_analyze['input_variables'].append(counter.get('input_variables', 0))
                stats_analyze['method_call'].append(counter.get('method_call', 0))
                stats_analyze['variable'].append(counter.get('variable', 0))

                distances.append(lev_distance)
                bleus.append(bleu_score * 100)
                print(str(idx) + ' \\ ' + str(len(loaded_obj)))
            except Exception as e:
                print(e)
                error += 1
        f.close()
        del loaded_obj



    files = str(args.pickle_files).split('/')
    # with open("stats.pickle", 'wb') as f:
    with open(os.path.join(files[0], "stats.pickle"), 'wb') as f:
        pickle.dump(stats_analyze, f)

    # plt.scatter(distances, bleus, s=10)
    # plt.savefig('hist.png')

    print(str(error) + ' files missing.')

    print('levenshtein distances:')
    print('mean:\t{}'.format(str(statistics.mean(distances))))
    print('median:\t{}'.format(str(statistics.median(distances))))
    print('std:\t{}'.format(str(statistics.pstdev(distances))))
    print('variance:\t{}'.format(str(statistics.pvariance(distances))))
    print('\nbleu scores:')
    print('mean:\t{}'.format(str(statistics.mean(bleus))))
    print('median:\t{}'.format(str(statistics.median(bleus))))
    print('std:\t{}'.format(str(statistics.pstdev(bleus))))
    print('variance:\t{}'.format(str(statistics.pvariance(bleus))))

    now = datetime.now()
    print("now =", now)
    # plt.show()
