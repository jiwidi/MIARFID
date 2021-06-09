
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("file.txt",header=None,names=["score","step","episode"])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(df["episode"], df["score"], label="Score")
plt.plot(df["episode"], df["score"].rolling(2).mean(), label="Rolling mean")
plt.legend(loc="upper left")
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("score_breakout.png")