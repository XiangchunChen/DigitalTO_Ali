# coding:utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
if __name__ == '__main__':


    x1=[1,2,3,6]
    y2= [30713/75.0 , 30713/75.0, 30713/75.0, 30713/75.0]
    y3= [  28649/75.0 ,18995  /75.0 ,14695/75.0 , 9927/75.0 ]
    y4= [29816/75.0 , 18526/75.0 , 17930/75.0  ,15516/75.0 ]
    y5= [28559/75.0 ,15574/75.0 ,  11552/75.0 ,7859/75.0 ]

    plt.figure(figsize=(32,29))
    plt.rc('font',family='Times New Roman')
    matplotlib.rcParams.update({'font.size': 100})
    width = 10
    l2=plt.plot(x1,y2,'g--',label='LE', linewidth=width)
    l3=plt.plot(x1,y3,'b--',label='Greedy', linewidth=width)
    l4=plt.plot(x1,y4,'b^-',label='DRL without DT', linewidth=width)
    l5=plt.plot(x1,y5,'ro-',label='DTDRLTO', linewidth=width)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)

    plt.xlabel('e) Bandwidth (Alibaba cluster-trace-v2018)')
    plt.ylabel('Average completion time (ms)')
    plt.xticks(x1)
    plt.legend()
    plt.savefig('bandwidth.pdf', dpi=120, bbox_inches='tight')
    plt.show()