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
    AllLogs[index] = AllLogs[index].strip()
    index += 1
#print(AllLogs)

#find scenes
settingRegex = "\[[\w\s]+\](?!\:)"

#scenes = firstEpisode[len(captainsLog):]
scenes = firstEpisode[firstEpisode.index("["):]
x = re.split(settingRegex, scenes)
x.remove("")
print(x[0])

#delete captainslog from scenes

#delete " " instead of \n delete unnecessary whitespace
#print(captainsLog)
with open('../Dataset/TNGScriptCut.json', 'w', encoding='utf-8') as json_file:
  json.dump(x, json_file)
