import tool.tools as ts
import csv

def readLParam(fn,lt=None):
    r = []
    if lt is None:
        with open(fn, 'r') as file:
            lst = list(csv.reader(file))
        print(len(lst[0]))
        print(lst)
        for i in range(10):
            if i < 6:
                r.append(float(lst[0][i]))
            else:
                r.append(lst[0][i])    
    return r

a = readLParam('./hyperpalam/li.csv')