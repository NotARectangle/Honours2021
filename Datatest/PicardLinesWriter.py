import json

character = "PICARD"
series = "TNG"


data = json.load(open('../Dataset/all_series_lines.json', 'r'))

character_lines = []
all_lines = {}
all_lines[character] = []
tng_episodes = data[series].keys()
#for ep in tng_episodes:

#    print(ep)
script = data[series]['episode 0']
character_lines = script[character]

for l in character_lines:
    all_lines[character].append(l)

with open('../Dataset/AllPicardLines2.json', 'w') as json_file:
  json.dump(all_lines, json_file)
