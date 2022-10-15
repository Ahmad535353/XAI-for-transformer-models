import string

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer_global = WordNetLemmatizer()

def get_attentions_of_this_index(attentions_lists, index):
    attentions_of_this_index = []
    for attention_layer in attentions_lists:
        attentions_of_this_index.append(attention_layer[index])
    return attentions_of_this_index


def get_attentions_of_ast_index(ast_index, mapped_indices, attentions_list):
    list_of_mapped_tokens = mapped_indices[ast_index]
    cumulative_attentions = []
    for attention_layer in attentions_list:
        attention_for_this_layer = 0
        for decoded_index in list_of_mapped_tokens:
            attention_for_this_layer += attention_layer[decoded_index]
        cumulative_attentions.append(attention_for_this_layer)
    return cumulative_attentions

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
    overlaps = []
    for code_tokens_sample, target_tokens_sample in zip(code_tokens_input, target_tokens_input):
        code_tokens_cleaned, target_tokens_cleaned = pre_process(code_tokens_sample, target_tokens_sample)
        counter = []
        for token in target_tokens_cleaned:
            # if any(token in x for x in code_tokens_sample):
            # changed it
            if any(token in x for x in code_tokens_cleaned):
                counter.append(token)
        overlaps.append(len(counter) / len(target_tokens_cleaned))
    return overlaps


def pre_process_nl(token):
    punctuations = [char for char in string.punctuation]
    if (token in punctuations) or (len(token) < 2):
        return 0
    return lemmatizer_global.lemmatize(token.lower())


def pre_process_code(code_toekn):
    if len(code_toekn) < 2:
        return 0
    return code_toekn.lower()
