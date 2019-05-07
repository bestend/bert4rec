import argparse
import json
import operator
import os
import pickle

from tqdm import tqdm

from variables import TOKEN_MASK, TOKEN_PAD, TOKEN_UNK, VALUE_PAD, VALUE_UNK, VALUE_MASK, DATA_SEQUENCE


def get_permit_token_ids(data_frequency, total_count, cut_ratio):
    sorted_frequency = sorted(data_frequency['token'].items(), key=operator.itemgetter(1), reverse=True)

    accum_count = 0
    permit_token_ids = []
    for srl, freq in sorted_frequency:
        accum_count += freq
        if accum_count / total_count > cut_ratio:
            break
        permit_token_ids.append(srl)

    return permit_token_ids


def process(columns, token_to_id, id_to_token):
    modified_tokens = [id_to_token[token] for token in columns['token']]
    modified_tokens = [token_to_id.get(token, VALUE_UNK) for token in modified_tokens]
    columns['token'] = modified_tokens
    return columns


def convert(input_dir, cut_ratio):
    with open(os.path.join(input_dir, 'data.pickle'), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(input_dir, 'dic.pickle'), "rb") as f:
        data_dict = pickle.load(f)
    with open(os.path.join(input_dir, 'frequency.pickle'), "rb") as f:
        data_frequency = pickle.load(f)

    total_behavior = sum(data_frequency['token'].values())

    id_to_token = {v: k for k, v in data_dict['token'].items()}

    # generate new token dict
    permit_token_ids = get_permit_token_ids(data_frequency, total_behavior, cut_ratio)
    new_token_dict = {
        TOKEN_PAD: VALUE_PAD,
        TOKEN_UNK: VALUE_UNK,
        TOKEN_MASK: VALUE_MASK,
    }
    for token_id in permit_token_ids:
        new_token_dict[token_id] = len(new_token_dict)

    original_token_num = len(data_dict['token'])
    reduced_token_num = len(new_token_dict)

    print('token removed => {} / {} ({}%)'.format(
        reduced_token_num,
        original_token_num,
        100 - reduced_token_num / original_token_num * 100.0
    ))

    new_data = []
    for columns in tqdm(data):
        new_data.append(process(columns, new_token_dict, id_to_token))

    data_dict['token'] = new_token_dict
    params = dict()
    params['input'] = [{'name': k, 'size': len(data_dict[k])} for k in DATA_SEQUENCE]
    return params, new_data, data_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--cut_ratio', default=1.0, type=float)

    conf = parser.parse_args()

    if conf.cut_ratio >= 1.0 or conf.cut_ratio <= 0:
        print("cut ratio must be 0.0 < value < 1.0")
        exit(1)

    os.makedirs(conf.output_dir, exist_ok=True)

    params, data, data_dict = convert(conf.input_dir, conf.cut_ratio)

    with open(os.path.join(conf.output_dir, 'data.pickle'), "wb") as f:
        pickle.dump(data, f)

    with open(os.path.join(conf.output_dir, 'dic.pickle'), "wb") as f:
        pickle.dump(data_dict, f)

    with open(os.path.join(conf.output_dir, 'params.json'), "w") as f:
        json.dump(params, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
