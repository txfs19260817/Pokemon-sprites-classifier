import csv


def generate_label_csv(classes):
    with open('../label.csv', 'w') as f:
        f.writelines('id,name\n')
        for i, c in enumerate(classes):
            f.writelines(str(i) + ',' + c + '\n')


def label_csv2dict(csv_path):
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[int(row['id'])] = row['name']
    return result
