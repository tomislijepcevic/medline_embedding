def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('groups')
    parser.add_argument('samples', type=int)
    parser.add_argument('dest')
    args = parser.parse_args()

    from os import path
    import yaml
    import pandas as pd

    from ..data import (
        read_test_texts,
        read_test_labels,
        read_classes
    )

    groups = yaml.load(open(args.groups))
    labels = read_test_labels()
    terms = read_classes()
    terms = [t.lower() for t in terms]

    for group, group_terms in groups.items():
        filepath = path.join(args.dest, group + '.csv')

        if path.isfile(filepath):
            continue

        linenos = dict()
        for term in group_terms:
            term_index = terms.index(term)
            term_linenos = labels[:, term_index].nonzero()[0]
            term_linenos = [l for l in term_linenos if l not in linenos]
            term_linenos = term_linenos[:args.samples]

            for lineno in term_linenos:
                linenos[lineno] = term

        targets = sorted(linenos.items(), key=lambda x: x[0])
        texts = read_test_texts()
        df = pd.DataFrame(columns=['term', 'text'])
        for lineno, text in enumerate(texts):
            if lineno == targets[0][0]:
                term = targets.pop(0)[1]
                df.loc[len(df)] = [term, text]

                if not targets:
                    break

        df.to_csv(filepath, index=False)


if __name__ == '__main__':
    main()
