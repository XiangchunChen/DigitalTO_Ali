# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
if __name__ == '__main__':


    x1=[25,50,75,100]
    # x2=[]

    y2= [19.64, 385.5, 409.51, 2745.75]
    y3= [21.04, 180.24, 245.21, 1059.34]
    y4= [42.72, 233.9, 253.27, 1213.6]
    y5= [17.04, 166.34, 207.65, 633.12]

    # x=np.arange(20,350)
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
    y = np.arange(500,2000,500, dtype=int)
    plt.xticks(x1)
    # plt.yticks(y)
    # plt.plot(x1,y1,'ro-')
    # plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
    # plt.title('Effect of changing number of tasks')
    plt.xlabel('a) Number of tasks (Alibaba cluster-trace-v2018)')
    plt.ylabel('Average completion time (ms)')
    plt.legend()
    plt.savefig('task.pdf', dpi=120, bbox_inches='tight')
    plt.show()