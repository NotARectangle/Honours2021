import json

character = "PICARD"
series = "TNG"


data = json.load(open('../Dataset/all_series_lines.json', 'r'))

character_lines = []
all_lines = {}
all_lines[character] = []
tng_episodes = data[series].keys()

for ep in tng_episodes:
    script = data[series][ep]
    character_lines = script[character]

    #for l in character_lines:
    all_lines[character].append(character_lines[0])

with open('../Dataset/AllPicardfirstLines.json', 'w') as json_file:
  json.dump(all_lines, json_file)