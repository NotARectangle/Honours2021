import json

series = "TNG"
data = json.load(open('../Dataset/all_scripts_raw.json', 'r'))

tNGScriptUncut = data[series]
episodes = tNGScriptUncut.keys()

messyScript = {};

#Maybe start reading from captains log
#print episode 1 script
#print(tNGScriptUncut['episode 0'])
#episode0 = tNGScriptUncut['episode 0']
#file = open("../Dataset/TNGEpisode0.txt", "w")
#file.write(episode0)
#file.close()
for ep in episodes:
    messyScript[ep]=tNGScriptUncut[ep]

with open('../Dataset/TNGScriptUncut.json', 'w') as json_file:
  json.dump(messyScript, json_file)
