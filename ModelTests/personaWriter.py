"""
Persona dataset should be fomatted like so:
{Persona ID: ["Picard", "Example senteces? or more information"],
    utterances: [history : {}, reply : {}] }
And add Bos token to history and eos token to end of reply
"""
import json
import re

data = json.load(open('../Dataset/TNGScriptCut2.json', 'r'))


persona = "PICARD"

#only permit conversations that include Picard
episodes = data.keys()
#newData = {}
utterances = []
for ep in episodes:
    episodeEntry = data[ep]
    scenes = episodeEntry["Scenes"]

    for scene in scenes:
        if persona in scene:
            #find the reply string
            replyIndex = scene.rindex(persona)
            reply = scene[replyIndex:]
            if "\n" in reply:
                reply = reply[:reply.index("\n")]
            reply = reply + " <eos>"
            history = scene[:replyIndex]
            #history = re.sub(r"(\([A-Za-z\s]+\))", "", history) do this while there is still that in history. 
            history = re.split("\\n", history)
            while "" in history:
                history.remove("")
            utterance = {"history" : history, "reply" : [reply]}
            utterances.append(utterance)

#hardcoding in some example introductions for now into Persona ID
newData = {"PersonaID" : ["<bos>", persona, "I'm Jean-Luc Picard, Captain of the Enterprise."], "utterances" : utterances}
print(len(newData["utterances"]))
with open('../Dataset/picardData.json', 'w', encoding='utf-8') as json_file:
  json.dump(newData, json_file)
