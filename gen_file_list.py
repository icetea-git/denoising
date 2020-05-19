import os
import json

PATH = '.\\LIDC-IDRI'


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if (len(files) > 3) & name.endswith('.dcm'):
                r.append(os.path.join(root, name))
    # random.shuffle(r)
    return r


with open('data.json', 'w') as outfile:
    json.dump(list_files(PATH), outfile)
