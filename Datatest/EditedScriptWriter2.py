# Author Milena Bromm
# Student ID 40325069
# Project Name: Honours 2021

import json
import re

data = json.load(open('../Dataset/TNGScriptUncut.json', 'r'))

firstEpisode = data['episode 0']

#break conversation on setting [setting] conv1...2
#take out stage direction (between brackets)

#Take out ending "<back"
episodes = data.keys()
newData = {}

for ep in episodes:
    EpisodeScript = data[ep]
    EpisodeScript = EpisodeScript[0: EpisodeScript.index("<Back")].strip()
    #Don't replace \n just make sure it is just one \n
    #EpisodeScript = EpisodeScript.replace("\n", " ")

    #get all captains Logs
    captainRe = "(Captain's log[^\[]+)"
    AllLogs = []
    AllLogs= re.findall(captainRe, EpisodeScript)
    index = 0
    #remove all captains logs
    #firstEpisode = firstEpisode.replace("Captain's log[^\[]+", " ")
    #strip trailing whitespace
    while index < len(AllLogs):
        EpisodeScript = EpisodeScript.replace(AllLogs[index], "")
        #make all whitespace instances just one space
        AllLogs[index] = re.sub(r"(\n)+", "\\n", AllLogs[index])
        #AllLogs[index] = re.sub(r"\s+", " ", AllLogs[index])
        AllLogs[index] = AllLogs[index].strip()
        index += 1




    #find scenes
    settingRegex = "\[[\w\s'-]+\](?!\:)"
    x = []
    #split scenes by setting notation ex. [Bridge]
    scenes = EpisodeScript[EpisodeScript.index("["):]
    x = re.split(settingRegex, scenes)
    #x.remove("")

    index = 0
    #delete trailing whitespace
    while index < len(x):
        #delete  [OC]-overcomms notation and whitespace before it
        x[index] = re.sub(r"\s\[[\w\s']+\]", "", x[index].strip())
        #make all whitespace instances just one space
        x[index] = re.sub(r"(\n){2,}", "\\n", x[index])
        #remove any \n inside one character speaking, only break before character
        x[index] = re.sub(r"(\n)(?!([A-Z]+ ?[A-Z]+ ?:)|[A-Z]:)(?!\()", " ", x[index])
        #x[index] = re.sub(r"\s+", " ", x[index])

        index += 1

    dict = {"Logs" : AllLogs, "Scenes" : x}
    newData[ep] = dict

with open('../Dataset/TNGScriptCut2.json', 'w', encoding='utf-8') as json_file:
  json.dump(newData, json_file)
