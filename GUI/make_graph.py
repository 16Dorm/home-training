import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from matplotlib import style

def make_graph():
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    style.use('ggplot')

    colors = ['red','yellowgreen']
    labels = ['incorrect', 'correct']
    ratio = [18, 92]
    explode = (0.2, 0.0)

    plt.pie(ratio, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14})
    plt.savefig('./GUI/graph.png', transparent=True)
    plt.show()

make_graph()