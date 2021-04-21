import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To make plots for who has the most lines in the series. Coded by following
# https://www.kaggle.com/zoiaran/who-has-the-most-words-and-lines-in-ds9

data = json.load(open('../Dataset/all_series_lines.json', 'r'))

tng_episodes = data['TNG'].keys()
total_word_counts={}
total_line_counts={}
#for i, ep in enumerate(episodes)
series = "TNG"
print(data['TNG']['episode 0'].keys())

for ep in tng_episodes:
    #for all characters in list
    script = data[series][ep]
    characters = data[series][ep].keys()
    for member in characters:
        character_lines = script[member]
        total_words_by_member_in_ep = 0
        total_lines_by_member_in_ep = 0
        #total_words_by_member = sum()
        for l in character_lines:
            total_words_by_member_in_ep += len(l.split())
            total_lines_by_member_in_ep += 1

        if member in total_word_counts.keys():
            total_word_counts[member]=total_word_counts[member]+total_words_by_member_in_ep
            total_line_counts[member]=total_line_counts[member]+total_lines_by_member_in_ep
        else:
            total_word_counts[member]=total_words_by_member_in_ep
            total_line_counts[member]=total_lines_by_member_in_ep

words_df=pd.DataFrame(list(total_word_counts.items()), columns=['Character','No. of Words'])
most_words=words_df.sort_values(by='No. of Words', ascending=False).head(12)
lines_df=pd.DataFrame(list(total_line_counts.items()), columns=['Character','No. of Lines'])
most_lines=lines_df.sort_values(by='No. of Lines', ascending=False).head(12)


most_words.plot.bar(x='Character',y='No. of Words')

most_lines.plot.bar(x='Character',y='No. of Lines')
plt.show()


