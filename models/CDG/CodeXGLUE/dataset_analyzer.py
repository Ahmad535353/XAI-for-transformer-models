import json
import math
import os
import pickle
import string

import numpy as np
from matplotlib import pyplot as plt

import bleu

import nltk
from nltk.stem import WordNetLemmatizer

from utils import calculate_the_overlap

test_file = '../dataset/python/test.jsonl'


def read_dataset(input_file):
    examples_code_tokens = []
    examples_target_tokens = []
    with open(input_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code_tokens = js['code_tokens']
            target_tokens = js['docstring_tokens']
            examples_code_tokens.append(code_tokens)
            examples_target_tokens.append(target_tokens)
    return examples_code_tokens, examples_target_tokens


# def clean_target_tokens(target_tokens):
#     punctuations = [char for char in string.punctuation]
#     target_tokens = set(tokens for tokens in target_tokens if tokens not in punctuations)
#     return target_tokens


# def pre_process(code_tokens_inp, target_tokens_inp):
#     lemmatizer = WordNetLemmatizer()
#
#     code_tokens_cleaned = []
#     for token in code_tokens_inp:
#         if len(token) < 2:
#             continue
#         temp = [lemmatizer.lemmatize(w.lower()) for w in token.split('_')]
#         code_tokens_cleaned += temp
#
#     punctuations = [char for char in string.punctuation]
#     target_tokens_cleaned = set()
#     for token in target_tokens_inp:
#         if token in punctuations:
#             continue
#         target_tokens_cleaned.add(lemmatizer.lemmatize(token.lower()))
#
#     return code_tokens_cleaned, target_tokens_cleaned


# def calculate_the_overlap(code_tokens_input, target_tokens_input):
#     overlaps = []
#     for code_tokens_sample, target_tokens_sample in zip(code_tokens_input, target_tokens_input):
#         code_tokens_cleaned, target_tokens_cleaned = pre_process(code_tokens_sample, target_tokens_sample)
#         counter = []
#         for token in target_tokens_cleaned:
#             if any(token in x for x in code_tokens_sample):
#                 counter.append(token)
#         overlaps.append(len(counter) / len(target_tokens_cleaned))
#     return overlaps


# m1 is the reference map
# m2 is the prediction map
def bleuFromMaps(m1, m2):
    scores = []
    num = 0.0

    for key in m1:
        if key in m2:
            bl = bleu.bleu(m1[key], m2[key][0])
            scores.append(bl)
            num += 1
    return scores


if __name__ == '__main__':

    code_tokens, target_tokens = read_dataset(test_file)

    nltk.download('wordnet')
    counts = calculate_the_overlap(code_tokens[:], target_tokens[:])

    predictions = []
    pickle_file = 'model/python/prediction_pickle.pickle'
    f = open(pickle_file, "rb")
    loaded_obj = pickle.load(f)
    for sample in loaded_obj:
        predictions.append(str(sample['idx']) + '\t' + sample['best_prediction'])
    (goldMap, predictionMap) = bleu.computeMaps(predictions, 'model/python/test_0.gold')
    bleus = bleuFromMaps(goldMap, predictionMap)

    counts_np = np.array(counts)
    bleus_np = np.array([x[0] for x in bleus])

    new_file = open("model/python/prediction_pickle_with_bleu_and_overlap.pickle", 'wb')
    for loaded_sample, bleu, overlap in zip(loaded_obj, bleus, counts):
        loaded_sample['bleu'] = bleu[0]
        loaded_sample['overlap'] = overlap
    pickle.dump(loaded_obj, new_file)


    high_bleu_threshold = 0.66
    low_bleu_threshold = 0.33
    high_overlap_threshold = 0.66
    low_overlap_threshold = 0.33

    count_group = [0, 0, 0, 0]
    for i in range(len(bleus_np)):
        if counts_np[i] >= high_overlap_threshold:  # These are "easy" samples
            if bleus_np[i] >= high_bleu_threshold:
                count_group[0] += 1
            elif bleus_np[i] <= low_bleu_threshold:
                count_group[1] += 1
        elif counts_np[i] <= low_overlap_threshold:  # These are "hard" samples
            if bleus_np[i] >= high_bleu_threshold:
                count_group[2] += 1
            elif bleus_np[i] <= low_bleu_threshold:
                count_group[3] += 1
    print(count_group)
    bins = np.linspace(0,
                       max(bleus_np),
                       40)  # fixed number of bins
    # bins = np.linspace(math.ceil(min(bleus_np)),
    #                    math.floor(max(bleus_np)),
    #                    20)  # fixed number of bins
    #
    plt.xlim([min(bleus_np), max(bleus_np)])

    plt.hist(bleus_np, bins=bins, alpha=0.5)
    plt.title('Smoothed Bleu-4 score')
    plt.xlabel('Bleu-4 score (40 evenly spaced bins)')
    plt.ylabel('count')

    plt.show()
