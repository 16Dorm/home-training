from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from matplotlib import style

def make_graph(wrong_frames, full_frames, num):

    wrong_per = round((wrong_frames/full_frames)*100, 2)
    correct_per = 100 - wrong_per

    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    style.use('ggplot')

    colors = ['red','yellowgreen']
    labels = ['incorrect', 'correct']
    ratio = [wrong_per, correct_per]
    explode = (0.2, 0.0)

    plt.figure(figsize=(4,4))
    plt.pie(ratio, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14})
    # plt.savefig('./GUI/graph_'+ str(num) +'.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.savefig('./play_results/graph_'+ str(num) +'.png', transparent=True, bbox_inches='tight', pad_inches=0)
    #plt.show()
