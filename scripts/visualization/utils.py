import csv

def load_labels(input_filepath):
    labels = []
    with open(input_filepath, 'r') as infile:
        for row in infile:
            labels.append(row.strip())
    return labels

def write_to_csv(output_filepath, rows, feature_labels, target_labels):
    with open(output_filepath, 'w') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow(['feature'] + target_labels)
        for row, fl in zip(rows, feature_labels):
            csvwriter.writerow([fl] + list(row))