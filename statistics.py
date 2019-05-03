import argparse

from sampler import read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    conf = parser.parse_args()

    data, _, params = read_data(conf.input_dir)

    '''
    빈도
    deal_count = defaultdict(lambda: 0)
    total_count = 0
    for e in tqdm(data):
        total_count += len(e['deal'])
        for d in e['deal']:
            deal_count[d] += 1

    sorted_by_count = sorted(deal_count.items(), key=operator.itemgetter(1), reverse=True)

    total_deal = len(deal_count)
    with open("statistics.txt", "w") as f:
        f.write("0\t0\n")
        accum_count = 0
        for idx, cc in enumerate(tqdm(sorted_by_count)):
            accum_count += cc[1]
            f.write("{}\t{}\n".format(idx / total_deal, accum_count / total_count))
    '''

    '''
    # 마지막 deal이 사용자의 이전 history에 포함 되어 있을 확률 28.69%
    hit_count = 0
    for e in tqdm(data):
        if e['deal'][-1] in e['deal'][:-1]:
            hit_count += 1
    print("hit = {}".format(hit_count / len(data) * 100.0))
    '''


if __name__ == '__main__':
    main()
