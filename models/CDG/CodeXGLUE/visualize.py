import json
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


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


# words = 'The quick brown fox jumps over the lazy dog'.split()
# color_array = np.random.rand(len(words))
# s = colorize(words, color_array)
#
# # to display in ipython notebook
# # from IPython.display import display, HTML
#
# # display(HTML(s))
#
# # or simply save in an html file and open in browser
# with open('colorize.html', 'w') as f:
#     f.write(s)

pickle_file = 'model/python/prediction_pickle_with_bleu_and_overlap.pickle'
model_type = 'roberta'
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
_, _, tokenizer_class = MODEL_CLASSES[model_type]
model_name_or_path = 'microsoft/codebert-base'
do_lower_case = False
if __name__ == '__main__':
    f = open(pickle_file, "rb")
    loaded_obj = pickle.load(f)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)

    j = 0
    high_bleu_threshold = 0.66
    low_bleu_threshold = 0.33
    high_overlap_threshold = 0.66
    low_overlap_threshold = 0.33

    categories = {'easy_good': [], 'easy_bad': [], 'hard_good': [], 'hard_bad': []}
    for sample in loaded_obj:
        # f = open('model/colorize' + str(j) + '.html', 'w')
        f = open('model/colorize/' + str(sample['idx']) + '.html', 'w', encoding='utf8')
        f.write('goal:')
        f.write('<br>')
        f.write(sample['target'])
        f.write('<br>')
        f.write('best prediction:')
        f.write('<br>')
        f.write(sample['best_prediction'])
        f.write('<br>')
        f.write('bleu score:\t')
        f.write("{:.5f}".format(sample['bleu']))
        f.write('<br>')
        f.write('overlap:\t')
        f.write("{:.5f}".format(sample['overlap']))
        f.write('<br><br><br>')

        if sample['overlap'] >= high_overlap_threshold:  # These are "easy" samples
            if sample['bleu'] >= high_bleu_threshold:
                categories['easy_good'].append(sample['idx'])
            elif sample['bleu'] <= low_bleu_threshold:
                categories['easy_bad'].append(sample['idx'])
        elif sample['overlap'] <= low_overlap_threshold:  # These are "hard" samples
            if sample['bleu'] >= high_bleu_threshold:
                categories['hard_good'].append(sample['idx'])
            elif sample['bleu'] <= low_bleu_threshold:
                categories['hard_bad'].append(sample['idx'])

        code = []
        # colorized = []
        for encoded_token in sample['source_code_tokens']:
            code.append(tokenizer.decode(encoded_token, clean_up_tokenization_spaces=False))
        code[0] = 'START'
        code[code.index('</s>')] = 'END'
        for attention_layer in sample['attentions']:     # number of decoder layers
            attention = np.array(attention_layer, dtype="float64")
            normalized_attention = ((attention - min(attention)) / (max(attention) - min(attention)))
            # colorized.append(colorize(code, normalized_attention))
            colorized = colorize(code, normalized_attention)
            f.write(colorized)
            f.write('<br><br><br>')
        f.close()
        print('wrote ' + str(sample['idx']))
        j += 1

    f = open('model/colorize/list.html', 'w', encoding='utf8')
    f.write(json.dumps(categories))

    print('hi')
