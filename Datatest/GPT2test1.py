import re
import json
import numpy as np
from sklearn.model_selection import train_test_split

data = json.load(open('../Dataset/AllPicardLines2.json', 'r'))

#print(data.keys())
def build_text_files(data_json, dest_path):
     f = open(dest_path, 'w', encoding='utf-8')
     data = ''
     for texts in data_json:
         summary = str(texts).strip()
         data += summary + " "
     f.write(data)

train, test = train_test_split(data['PICARD'],test_size=0.15)




build_text_files(train,'train_dataset.txt')
build_text_files(test,'test_dataset.txt')

print("Train dataset length: "+str(len(train)))
print("Test dataset length: "+ str(len(test)))




#DO the same but try to model dialog instead.