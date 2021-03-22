import csv

# crop parameters
expected_size = (1280, 720)
begin = (85, 20)
box = (75, 75)
offset_x, offset_y = 588, 186


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


def resize_and_crop(image):
    if image.width != 1280 and image.height != 720:
        image = image.resize(expected_size)
    crop_list = [
        image.crop((begin[0], begin[1], begin[0] + box[0], begin[1] + box[1])),
        image.crop((begin[0] + offset_x, begin[1], begin[0] + offset_x + box[0], begin[1] + box[1])),
        image.crop((begin[0], begin[1] + offset_y, begin[0] + box[0], begin[1] + offset_y + box[1])),
        image.crop(
            (begin[0] + offset_x, begin[1] + offset_y, begin[0] + offset_x + box[0], begin[1] + offset_y + box[1])),
        image.crop((begin[0], begin[1] + offset_y * 2, begin[0] + box[0], begin[1] + offset_y * 2 + box[1])),
        image.crop((begin[0] + offset_x, begin[1] + offset_y * 2, begin[0] + offset_x + box[0],
                    begin[1] + offset_y * 2 + box[1])),
    ]
    return crop_list
