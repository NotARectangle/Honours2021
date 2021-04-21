# Author Milena Bromm
# Student ID 40325069
# Project Name: Honours 2021
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = json.load(open('../Dataset/tngPersonaData.json', 'r'))

keys = data.keys()

ydata = {}

for persona in keys:
    print(persona)
    ydata[persona] = len(data[persona]["utterances"])

words_df=pd.DataFrame(list(ydata.items()), columns=['Character','No. of Conversations'])
most_words=words_df.sort_values(by='No. of Conversations', ascending=False).head(12)


most_words.plot.bar(x='Character',y='No. of Conversations')

plt.show()
