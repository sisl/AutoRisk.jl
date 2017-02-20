import csv

def load_labels(input_filepath):
    labels = []
    with open(input_filepath, 'r') as infile:
        for row in infile:
            elements = row.strip().split(' ')
            if '*' in elements:
                repeat = int(elements[-1])
                base = elements[0] + '_{}'
                for r in range(1, repeat + 1):
                    labels.append(base.format(r))
            else:
                labels.append(row.strip())
    return labels

def write_to_csv(output_filepath, rows, feature_labels, target_labels):
    with open(output_filepath, 'w') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow(['feature'] + target_labels)
        for row, fl in zip(rows, feature_labels):
            csvwriter.writerow([fl] + list(row))