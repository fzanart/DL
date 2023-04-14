import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

df = pd.read_csv('/home/panchoz/Documents/github/data/GHDomains.csv')

######## for comparable dataset ########
df.dropna(subset=['Readme'], inplace=True)
df.drop(31, axis=0, inplace=True)   #removed repo Homebrew/legacy-homebrew      # Software tools
df.drop(124, axis=0, inplace=True)  #removed repo shadowsocks/shadowsocks       # Software tools
df.drop(237, axis=0, inplace=True)  #removed repo npm/npm                       # Software tools
df.drop(3057, axis=0, inplace=True) #removed repo firstopinion/formatter.js     # Web libraries and frameworks
df.drop(4488, axis=0, inplace=True) #removed repo jersey/jersey                 # Web libraries and frameworks
df.reset_index(inplace=True, drop=True)


text = df['clean_description'] + df['clean_readme']
test = text.astype(str)
y = df['Domain']
######## for comparable dataset ########


######## last iteration ########
# # drop unreachable repos
# df = df[df['Status']==True].reset_index(drop=True)
# text = df['clean_description'] + df['clean_readme']

# # drop empty descp + readme text repos
# to_drop = []
# for i in text.index:
#     if not isinstance(text.loc[i], str):
#         to_drop.append(i)

# text = text.drop(to_drop).reset_index(drop=True)
# y = df['Domain'].drop(to_drop).reset_index(drop=True)
######## last iteration ########

# encode text
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text)

# split data
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.1, random_state=42, stratify=y)


np.save('/home/panchoz/Documents/github/data/X_train_embeddings',X_train)
np.save('/home/panchoz/Documents/github/data/X_test_embeddings', X_test)

y_train.to_csv('/home/panchoz/Documents/github/data/y_train.csv')
y_test.to_csv('/home/panchoz/Documents/github/data/y_test.csv')