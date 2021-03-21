def generate_label_csv(classes):
    with open('label.csv', 'w') as f:
        f.writelines('id,name\n')
        for i, c in enumerate(classes):
            f.writelines(str(i)+','+c+'\n')