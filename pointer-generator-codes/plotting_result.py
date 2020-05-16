import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("rouge1.csv")

sns.barplot(x="sentence_len", y="repetition", hue="weight", data=data)
plt.xlabel("Input Sentence", fontsize=15)
plt.ylabel("Repetition Percentage", fontsize=15)
plt.title("Bar Plot of Repetition Percentage", fontsize=18)
plt.legend(title="Cov_weight", fontsize=12, title_fontsize=13, loc="upper right")
plt.xticks(ticks=[0, 1, 2, 3], labels=["N", "3", "6", "10"])
plt.show()

sns.barplot(x="sentence_len", y="rouge1", hue="weight", data=data)
plt.xlabel("Input Sentence", fontsize=15)
plt.ylabel("Repetition Percentage", fontsize=15)
plt.title("Bar Plot of Rouge1 Score", fontsize=18)
plt.legend(title="Cov_weight", fontsize=12, title_fontsize=13, loc="upper right")
plt.xticks(ticks=[0, 1, 2, 3], labels=["N", "3", "6", "10"])
plt.show()