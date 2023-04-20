import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

df = pd.read_csv('/home/panchoz/Documents/github/data/GHDomains.csv')

# drop unreachable repos
df = df[df['Status']==True].reset_index(drop=True)
text = df['clean_description'] + df['clean_readme']

# drop empty descp + readme text repos
to_drop = []
for i in text.index:
    if not isinstance(text.loc[i], str):
        to_drop.append(i)

text = text.drop(to_drop).reset_index(drop=True)
y = df['Domain'].drop(to_drop).reset_index(drop=True)

# split data
X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=42, stratify=y)

# encode text
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(X_train)
np.save('/home/panchoz/Documents/github/data/X_train_embeddings',embeddings)

embeddings = model.encode(X_test)
np.save('/home/panchoz/Documents/github/data/X_test_embeddings', embeddings)

y_train.to_csv('/home/panchoz/Documents/github/data/y_train.csv')
y_test.to_csv('/home/panchoz/Documents/github/data/y_test.csv')