import json
import re

data = json.load(open('../Dataset/TNGScriptUncut.json', 'r'))

firstEpisode = data['episode 0']

#break conversation on setting [setting] conv1...2
#take out stage direction (between brackets)

#Take out ending "<back"

firstEpisode = firstEpisode[0: firstEpisode.index("<Back")].strip()

firstEpisode = firstEpisode.replace("\n", " ")
#leav out episode name? start with captains log.. Episode name probably not important data
captainsLog = firstEpisode[firstEpisode.index("Captain's log"):firstEpisode.index("[")].strip()

#get all captains Logs
captainRe = "(Captain's log[^\[]+)"
AllLogs= re.findall(captainRe, firstEpisode)
index = 0
#remove all captains logs
#firstEpisode = firstEpisode.replace("Captain's log[^\[]+", " ")
#strip trailing whitespace
while index < len(AllLogs):
    firstEpisode = firstEpisode.replace(AllLogs[index], " ")
    #make all whitespace instances just one space
    AllLogs[index] = re.sub(r"\s+", " ", AllLogs[index])
    AllLogs[index] = AllLogs[index].strip()
    index += 1
#print(AllLogs)

#find scenes
settingRegex = "\[[\w\s']+\](?!\:)"

#split scenes by setting notation ex. [Bridge]
scenes = firstEpisode[firstEpisode.index("["):]
x = re.split(settingRegex, scenes)
x.remove("")

index = 0
#delete trailing whitespace
while index < len(x):
    #delete  [OC]-overcomms notation
    x[index] = re.sub(r"\[[\w\s']+\]", "", x[index].strip())
    #make all whitespace instances just one space
    x[index] = re.sub(r"\s+", " ", x[index])
    index += 1

dict = {"Logs" : AllLogs, "Scenes" : x}

with open('../Dataset/TNGScriptCutEp0.json', 'w', encoding='utf-8') as json_file:
  json.dump(dict, json_file)
