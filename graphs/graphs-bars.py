import matplotlib.pyplot as plt

import numpy as _np

states = ['625', '1000','625','1024','729', '800', '512', '512', '1024',
          '4900', '4000', '2500', '3125', '2000',  '2048', '1944', '3888', '3456',
          '10000', '8000', '7000', '7000', '4096', '5184', '4374', '5832', '5184',
          '14400' ,'12500', '10000', '10000', '7560', '9216', '6561', '8748', '7776',
          '19600', '18750', '12000', '12500' ,'11250' ,'10368', '9072', '9216', '8192'
          ]
values_tensor = [0.48,	1.24,	1.30,	2.74,	2.877,	5.029,	4.27,	5.45,	12.85,
                 5.84,	7.67,	6.20,	11.28,	10.29,	13.26,	17.06,	44.63,	49.27,
                 16.11,	15.23,	18.39,	26.09,	20.54,	34.74,	37.96	,66.21	,74.37,
                 31.56,	26.93,	26.38,	37.61,	38.62,	62.89,	57.67	,99.17	,111.92,
                 42.75,	40.93,	32.95,	46.68,	57.14,	71.75,	81.24	,106.02	,117.92]
values_tabular = [0.28,	0.98,	0.94,	2.56	,2.54,	4.15	,3.45,	4.48	,11.79,
                  13.12,	9.88	,8.299,	15.88,	16.76,	23.35	,25.31	,80.75,	87.62,
                  51.35,	34.82	,31.65,	44.55	,31.45,	58.79	,62.51,	123.22	,137.39,
                  123.40	,66.93,	65.98	,96.79,	100.13,	169.30,	147.247	,299.27	,333.43,
                  253.75,	109.49	,83.46,	128.32,	156.21,	190.59	,178.35,	307.20	,334.96]

#'22500', '90000', '250000', '640000', '1000000',
#'24000', '60000', '125000', '512000', '1000000',
#'22500', '67500', '160000', '540000', '1200000',
#'19200', '100000', '200000', '600000', '1200000',
#'27000', '86436', '262144', '531441', '1000000',

states = ['625', '4900', '10000', '14400', '19600',
          '1000', '4000', '8000', '12500', '18750',
          '625', '2500', '7000', '10000', '12000',
          '1024', '3125', '7000', '10000','12500',
          '729', '2000', '4096', '7560', '11250',
          '800', '2048', '5184', '9216', '10368',
          '512', '1944', '4374', '6561', '9072',
          '512', '3888', '5832', '8748', '9216',
          '1024', '3456', '5184', '7776', '8192'
          ]
values_tensor = [0.48, 5.84, 16.11,31.56, 42.75,
                 1.24,7.67, 15.23, 26.93,40.93,
                 1.30, 6.20,18.39, 26.38, 32.95,
                 2.74, 11.28, 26.09,37.61, 46.68,
                 2.87, 10.29,20.54, 38.62, 57.14,
                 5.02, 13.26, 34.74,62.89, 71.75,
                 4.27,17.06, 37.96, 57.67,81.24,
                 5.45, 44.63,66.21, 99.17, 106.02,
                 12.85, 49.27, 74.37,111.92, 117.92]
values_tabular = [0.28, 13.12, 51.35,123.40, 253.75,
                  0.98,9.88, 34.82, 66.93,109.49,
                  0.94, 8.29,31.65, 65.98, 83.46,
                  2.56, 15.88, 44.55,96.79, 128.32,
                  2.54,16.76, 31.45, 100.13,156.21,
                  4.15, 23.35,58.79, 169.30, 190.59,
                  3.45, 25.31, 62.51,147.24, 178.35,
                  4.48,80.75, 123.22, 299.27,307.20,
                  11.79, 87.62,137.39, 333.43, 334.96]

x = _np.arange(len(states))
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_size_inches(15,5)
rects1 = ax.bar(x - width/2, values_tensor, width, label='CP-MDP',#label=['CP-MDP 2D','CP-MDP 3D','CP-MDP 4D','CP-MDP 5D','CP-MDP 6D',
                                                          #'CP-MDP 7D','CP-MDP 8D','CP-MDP 9D','CP-MDP 10D'],
                                                                  color=['black', 'black','black','black','black',
                                                                          'red','red','red','red','red',
                                                                          'green', 'green', 'green', 'green', 'green',
                                                                          'blue', 'blue', 'blue', 'blue', 'blue',
                                                                          'gold','gold','gold','gold','gold',
                                                                          'cyan', 'cyan','cyan','cyan','cyan',
                                                                          'darkorange',  'darkorange', 'darkorange', 'darkorange', 'darkorange',
                                                                          'darkmagenta','darkmagenta','darkmagenta','darkmagenta','darkmagenta',
                                                                          'deeppink', 'deeppink', 'deeppink', 'deeppink', 'deeppink'])

rects2 = ax.bar(x + width/2, values_tabular, width, label='Tabular',#label=['Tabular 2D','Tabular 3D','Tabular 4D','Tabular 5D','Tabular 6D',
                                                           #'Tabular 7D','Tabular 8D','Tabular 9D','Tabular 10D'],
                                                                    color=['gray', 'gray','gray','gray','gray',
                                                                           'salmon', 'salmon', 'salmon', 'salmon', 'salmon',
                                                                           'lightgreen', 'lightgreen','lightgreen','lightgreen','lightgreen',
                                                                           'cornflowerblue', 'cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue',
                                                                           'lightyellow','lightyellow','lightyellow','lightyellow','lightyellow',
                                                                           'lightcyan', 'lightcyan', 'lightcyan', 'lightcyan', 'lightcyan',
                                                                           'peachpuff', 'peachpuff', 'peachpuff', 'peachpuff', 'peachpuff',
                                                                           'orchid', 'orchid', 'orchid', 'orchid', 'orchid',
                                                                           'lightpink','lightpink','lightpink','lightpink','lightpink'],
                                                                    edgecolor=['black', 'black','black','black','black',
                                                                          'red','red','red','red','red',
                                                                          'green', 'green', 'green', 'green', 'green',
                                                                          'blue', 'blue', 'blue', 'blue', 'blue',
                                                                          'gold','gold','gold','gold','gold',
                                                                          'cyan', 'cyan','cyan','cyan','cyan',
                                                                          'darkorange',  'darkorange', 'darkorange', 'darkorange', 'darkorange',
                                                                          'darkmagenta','darkmagenta','darkmagenta','darkmagenta','darkmagenta',
                                                                          'deeppink', 'deeppink', 'deeppink', 'deeppink', 'deeppink'],  hatch='//')

ax.set_ylabel('runtime (seconds)')
ax.set_xlabel('number of states')
plt.xticks(rotation=45)
ax.set_title('2, 3, 4, 5, 6, 7, 8, 9 and 10-D grids')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
#fig.savefig('graphs/runtime.pdf')




#'22500', '90000', '250000', '640000', '1000000',
#'24000', '60000', '125000', '512000', '1000000',
#'22500', '67500', '160000', '540000', '1200000',
#'19200', '100000', '200000', '600000', '1200000',
#'27000', '86436', '262144', '531441', '1000000',

states = ['625', '4900', '10000', '14400', '19600',
          '1000', '4000', '8000', '12500', '18750',
          '625', '2500', '7000', '10000', '12000',
          '1024', '3125', '7000', '10000','12500',
          '729', '2000', '4096', '7560', '11250',
          '800', '2048', '5184', '9216', '10368',
          '512', '1944', '4374', '6561', '9072',
          '512', '3888', '5832', '8748', '9216',
          '1024', '3456', '5184', '7776', '8192'
          ]
values_tensor = [131.54, 133.99, 141.66,145.11, 151.33,
                 132.72,133.79, 138.29, 142.94,151.02,
                 137.10, 133.44,141.55, 147.84, 152.23,
                 132.24, 139.22, 150.68,160.07, 168.42,
                 132.14, 137.02,147.96, 164.57,178.65,
                 133.98, 141.17, 161.14,180.65, 194.09,
                 133.47,144.69, 161.37,177.91,195.32,
                 133.89,169.24,189.78, 218.82, 226.70,
                 143.99,171.68, 191.39,222.72,229.97]
values_tabular = [160.66, 902.71, 3336.01,6780.85, 12453.95,
                  174.11,894.29, 3202.27, 7636.52,10941.84,
                  150.40, 525.78,3265.82, 6533.56, 9353.32,
                  209.92, 907.58, 4050.28,8135.09, 12639.80,
                  176.78,1530.37, 1737.94,5618.70,12289.15,
                  197.11, 595.76,3138.97, 9648.82, 12178.07,
                  159.46, 610.07, 2577.43,5642.09, 10671.69,
                  163.33,2304.49, 5029.51, 11157.96,12368.83,
                  293.93,2039.08,4430.59, 9810.34,10874.18]

x = _np.arange(len(states))
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_size_inches(15,5)
rects1 = ax.bar(x - width/2, values_tensor, width, label='CP-MDP',#label=['CP-MDP 2D','CP-MDP 3D','CP-MDP 4D','CP-MDP 5D','CP-MDP 6D',
                                                          #'CP-MDP 7D','CP-MDP 8D','CP-MDP 9D','CP-MDP 10D'],
                                                                  color=['black', 'black','black','black','black',
                                                                          'red','red','red','red','red',
                                                                          'green', 'green', 'green', 'green', 'green',
                                                                          'blue', 'blue', 'blue', 'blue', 'blue',
                                                                          'gold','gold','gold','gold','gold',
                                                                          'cyan', 'cyan','cyan','cyan','cyan',
                                                                          'darkorange',  'darkorange', 'darkorange', 'darkorange', 'darkorange',
                                                                          'darkmagenta','darkmagenta','darkmagenta','darkmagenta','darkmagenta',
                                                                          'deeppink', 'deeppink', 'deeppink', 'deeppink', 'deeppink'])

rects2 = ax.bar(x + width/2, values_tabular, width, label='Tabular',#label=['Tabular 2D','Tabular 3D','Tabular 4D','Tabular 5D','Tabular 6D',
                                                           #'Tabular 7D','Tabular 8D','Tabular 9D','Tabular 10D'],
                                                                    color=['gray', 'gray','gray','gray','gray',
                                                                           'salmon', 'salmon', 'salmon', 'salmon', 'salmon',
                                                                           'lightgreen', 'lightgreen','lightgreen','lightgreen','lightgreen',
                                                                           'cornflowerblue', 'cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue',
                                                                           'lightyellow','lightyellow','lightyellow','lightyellow','lightyellow',
                                                                           'lightcyan', 'lightcyan', 'lightcyan', 'lightcyan', 'lightcyan',
                                                                           'peachpuff', 'peachpuff', 'peachpuff', 'peachpuff', 'peachpuff',
                                                                           'orchid', 'orchid', 'orchid', 'orchid', 'orchid',
                                                                           'lightpink','lightpink','lightpink','lightpink','lightpink'],
                                                                    edgecolor=['black', 'black','black','black','black',
                                                                          'red','red','red','red','red',
                                                                          'green', 'green', 'green', 'green', 'green',
                                                                          'blue', 'blue', 'blue', 'blue', 'blue',
                                                                          'gold','gold','gold','gold','gold',
                                                                          'cyan', 'cyan','cyan','cyan','cyan',
                                                                          'darkorange',  'darkorange', 'darkorange', 'darkorange', 'darkorange',
                                                                          'darkmagenta','darkmagenta','darkmagenta','darkmagenta','darkmagenta',
                                                                          'deeppink', 'deeppink', 'deeppink', 'deeppink', 'deeppink'],  hatch='//')

ax.set_ylabel('memory (MB)')
ax.set_xlabel('number of states')
plt.xticks(rotation=45)
ax.set_title('2, 3, 4, 5, 6, 7, 8, 9 and 10-D grids')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

#fig.savefig('graphs/memory.pdf')





import matplotlib.pyplot as plt
import numpy as _np


states = [22500	,	24000	,	22500	,	19200	,	27000	,
          90000	,	60000	,	67500	,	100000,		86436	,
          250000	,	125000	,	160000	,	200000	,	229376	,
          640000	,	512000	,	540000,		600000	,	531441	,
          1000000	,	1000000	,	1200000	,	1200000	,	1000000
          ]
values_runtime_tensor = [154.85, 157.41, 170.67,182.74, 256.35,
                      246.07, 211.53, 253.23, 468.50,512.05,
                      458.64,302.22, 475.77, 758.86, 1261.16,
                      978.44, 975.83, 1287.93,2040.23, 2434.99,
                      1455.18, 2026.12,2607.35,3688.69,5770.65,
                 ]

x = _np.arange(len(states))
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_size_inches(8,5)
rects1 = ax.bar(x - width/2, values_runtime_tensor, width, label='Memory CP-MDP',
                                                                  color=['black',
                                                                          'red',
                                                                          'green',
                                                                          'blue',
                                                                          'gold'])



ax.set_ylabel('memory (MB)')
ax.set_xlabel('number of states')
plt.xticks(rotation=90)
ax.set_title('2, 3, 4, 5 and 6-D grids')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


autolabel(rects1)

fig.tight_layout()

plt.show()

#fig.savefig('graphs/memory-cp.pdf')



import matplotlib.pyplot as plt
import numpy as _np


states = [22500	,	24000	,	22500	,	19200	,	27000	,
          90000	,	60000	,	67500	,	100000,		86436	,
          250000	,	125000	,	160000	,	200000	,	229376	,
          640000	,	512000	,	540000,		600000	,	531441	,
          1000000	,	1000000	,	1200000	,	1200000	,	1000000
          ]
values_runtime_tensor = [48.26, 51.69, 64.85,75.39, 145.34,
                 215.35, 149.88, 238.86, 438.66,508.58,
                 586.95, 327.92, 531.71, 855.01, 1299.45,
                 1573.36, 1933.44, 2316.18,2794.62, 3442.86,
                 2422.61, 4598.49,5700.30,5886.99, 5903.21,
                 ]

x = _np.arange(len(states))
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_size_inches(8,5)
rects1 = ax.bar(x - width/2, values_runtime_tensor, width, label='Runtime CP-MDP',
                                                                  color=['black',
                                                                          'red',
                                                                          'green',
                                                                          'blue',
                                                                          'gold'])



ax.set_ylabel('runtime (seconds)')
ax.set_xlabel('number of states')
plt.xticks(rotation=90)
ax.set_title('2, 3, 4, 5 and 6-D grids')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


autolabel(rects1)

fig.tight_layout()

plt.show()

#fig.savefig('graphs/runtime-cp.pdf')