import os.path
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_path", help="path to the pickle file that we want to sample")
    args = parser.parse_args()

    f = open(args.pickle_path, "rb")
    loaded_obj = pickle.load(f)

    with open('sampled_pickle.pickle', 'wb') as f1:
        pickle.dump(loaded_obj[:100], f1)
    print('done')
