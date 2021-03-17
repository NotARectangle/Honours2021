"""
Persona dataset should be fomatted like so:
{Persona ID: ["Picard", "Example senteces? or more information"],
    utterances: [history : {}, reply : {}] }
And add Bos token to history and eos token to end of reply
"""
import json
import re

data = json.load(open('../Dataset/TNGScriptCut2.json', 'r'))


persona = "PICARD:"

#only permit conversations that include Picard
episodes = data.keys()
#newData = {}
utterances = []
for ep in episodes:
    episodeEntry = data[ep]
    scenes = episodeEntry["Scenes"]

    for scene in scenes:
        if persona in scene:
            repeat = True
        #    While repeat:
            #find the reply string
            replyIndex = scene.rindex(persona)
            reply = scene[replyIndex:]
            if "\n" in reply:
                reply = reply[:reply.index("\n")]

            reply = re.sub(r"\n?(\([\s\w,\.!'?/\\-]+\))", "", reply)
            history = scene[:replyIndex]
            history = re.sub(r"\n?(\([\s\w,\.!'?/\\-]+\))", "", history)
            #Check how long the the scene is
            scene = history + reply
            added_scenes = []
            pot_scenes = [scene]
            while repeat:
                if len(added_scenes) > 0:
                    pot_scenes = added_scenes
                    added_scenes = []
                for sc in pot_scenes:
                    scene_len = len(sc)
                    if scene_len > 400:
                        #split scene
                        persona_matches = re.finditer(persona+":", sc)
                        l_ind = []
                        for m in persona_matches:
                            l_ind.append(m.end())
                        #If persona appears more than 3 times split in different scenes
                        if len(l_ind) > 3: #split in half
                            half = int(len(l_ind) / 2)
                            split_index = sc.find("\n", l_ind[half])
                            split_scenes = [sc[:split_index], sc[split_index-1:]]
                            added_scenes.append(split_scenes[0])
                            added_scenes.append(split_scenes[1])
                        #if more than 400 words and persona does not appear that often,
                        else:
                        #only take 5 previous conversations, to persona talking into, consideration
                            speaker_re = "([A-Z]+ ?[A-Z]+ ?:)|[A-Z]:"
                            speaker_turns = re.finditer(speaker_re, sc)
                            f_ind = []
                            for m in speaker_turns:
                                f_ind.append(m.start())
                            #go backwards through string to make sure persona is last speaker.
                            num_speaker = len(f_ind)
                            count = num_speaker-1
                            p_is_last = False
                   #         sc_changed = False
                            while count > 0 and p_is_last is False:
                                last_sequence = sc[f_ind[count]:]
                                if re.match(persona, last_sequence):
                                    p_is_last = True
                                else:
                                    # adjust string
                                    sc = sc[:f_ind[count]]
                                    #remove last element if last element is not Persona
                                    f_ind.pop()
                                count -= 1

                            #if still more than 400 proceed
                            if len(sc) > 400:
                                num_speaker = len(f_ind)
                                if num_speaker > 5:
                                    f_ind = f_ind[-5:]
                                    added_scenes.append(sc[f_ind[0]:])
                                #if it is still to long
                                else:
                                    #at least three conversation turns If I can't cut the word length further, admit them for now
                                    if len(f_ind) > 3:
                                        f_ind = f_ind[1:]
                                        added_scenes.append(sc[f_ind[0]:])
                            else:
                                added_scenes.append(sc)

                if not (len(added_scenes) >0):
                    repeat = False

            for sc in pot_scenes:
                #sep reply and hist again
                str_sc = str(sc)
                replyIndex = str_sc.rindex(persona)
                reply = str_sc[replyIndex:]
                history = str_sc[:replyIndex]
                reply = reply + " <eos>"
                history = re.split("\\n", history)
                while "" in history:
                    history.remove("")

                utterance = {"history" : history, "reply" : [reply]}
                utterances.append(utterance)

#hardcoding in some example introductions for now into Persona ID
newData = {"PersonaID" : ["<bos>" + persona + " I am Jean-Luc Picard, Captain of the Enterprise."], "utterances" : utterances}
print(len(newData["utterances"]))
with open('../Dataset/picardData2.json', 'w', encoding='utf-8') as json_file:
  json.dump(newData, json_file)
