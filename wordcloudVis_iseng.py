from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import random
from palettable.colorbrewer.sequential import Greys_9
from palettable.colorbrewer.sequential import Reds_9
from palettable.colorbrewer.sequential import Blues_9
from palettable.colorbrewer.sequential import Greens_9
import re

font_path = "font/Mohave.otf"
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 50)

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Reds_9.colors[random.randint(4,8)])

# csv_path = "label_visualisasi/ahokdjarotfixedit.csv"
df = pd.read_csv('datasetFinal/dataset3class_V5fixxVis.csv', sep=',')
df = df[df.is_kelas == 2]
# patt1 = re.compile('(\s*)jokowi(\s*)')
words_array = df.text
# words = words_array.to_string()
str1 = ' '.join(str(e) for e in words_array)
str1 = str1.replace('jokowi', '')
# print(str1)
# with open("Output.txt", "w") as text_file:
#     text_file.write(words)

wc = WordCloud(font_path=font_path, background_color="white", max_words=150, max_font_size=300, width=800, height=400)
wc.generate(str1)
wc.recolor(color_func=color_func)

plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.savefig('class2.png', bbox_inches='tight')
plt.show()