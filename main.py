# -*- coding: utf-8 -*-

import sys

############### Read Original Data ###############
import pandas as pd
import numpy as np
# Read data from a csv
data = pd.read_csv('DatosOrdenados.csv', header=0, index_col=0)
# Order data
z = np.array(data.values)
x_size, y_size = z.shape
# Get ranges
x_range = np.arange(x_size)
y_range = np.arange(y_size)
# Get grid type values
X, Y = np.meshgrid(x_range, y_range)
Z = z[X,Y]
# Get labels
x_labels, y_labels = data.axes

############### Generate Aproximation Data ###############
#from numpy import sin, cos, exp
m1=0;
m2=0.5;
m3=1;

m4=0;
m5=0.5;
m6=1;

de1=4.25;
de2=3.25;
de3=1.25;

de4=0.25;
de5=0.25;
de6=0.25;

p1=0;
p2=0;
p3=0;
p4=100;
p5=0;
p6=0;
p7=0;
p8=0;
p9=0;

q1=0;
q2=0;
q3=0;
q4=0;
q5=100;
q6=0;
q7=0;
q8=0;
q9=0;

r1=0;
r2=0;
r3=0;
r4=10;
r5=0;
r6=0;
r7=0;
r8=0;
r9=0;

points = np.zeros(z.shape)

for x in x_range:
    x_temp = x / x_range[-1];
    for y in y_range:
        y_temp = y / y_range[-1];
        
        mf1 = np.exp((-(x_temp-m1)**2)/(2*de1**2))
        mf2 = np.exp((-(x_temp-m2)**2)/(2*de2**2))
        mf3 = np.exp((-(x_temp-m3)**2)/(2*de3**2))
        mf4 = np.exp((-(y_temp-m4)**2)/(2*de4**2))
        mf5 = np.exp((-(y_temp-m5)**2)/(2*de5**2))
        mf6 = np.exp((-(y_temp-m6)**2)/(2*de6**2))
          
        inf1 = mf1 * mf4
        inf2 = mf1 * mf5
        inf3 = mf1 * mf6
        inf4 = mf2 * mf4
        inf5 = mf2 * mf5
        inf6 = mf2 * mf6
        inf7 = mf3 * mf4
        inf8 = mf3 * mf5
        inf9 = mf3 * mf6
          
        reg1 = inf1 * (p1 * x_temp + q1 * y_temp + r1)
        reg2 = inf2 * (p2 * x_temp + q2 * y_temp + r2)
        reg3 = inf3 * (p3 * x_temp + q3 * y_temp + r3)
        reg4 = inf4 * (p4 * x_temp + q4 * y_temp + r4)
        reg5 = inf5 * (p5 * x_temp + q5 * y_temp + r5)
        reg6 = inf6 * (p6 * x_temp + q6 * y_temp + r6)
        reg7 = inf7 * (p7 * x_temp + q7 * y_temp + r7)
        reg8 = inf8 * (p8 * x_temp + q8 * y_temp + r8)
        reg9 = inf9 * (p9 * x_temp + q9 * y_temp + r9)
          
        b = inf1 + inf2 + inf3 + inf4 + inf5 + inf6 + inf7 + inf8 + inf9
        a = reg1 + reg2 + reg3 + reg4 + reg5 + reg6 + reg7 + reg8 + reg9
        
        points[x][y] = (a / b)

Z_aprox = points[X,Y]

############### Plot Data ###############
use_plotly = False
use_inline = False

if use_plotly:
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Surface(z=Z_aprox, x=X, y=Y, colorscale="inferno", opacity=0.5),
        go.Surface(z=Z, x=X, y=Y, colorscale="viridis", opacity=1)
        ])
    fig.update_layout(title = 'Ocupation',
                      autosize = True,
                      margin = dict(l=65, r=50, b=65, t=90),
                      scene = dict(
                          xaxis = dict(
                              showticklabels = False,
                              ticktext = x_labels,
                              tickvals = x_range,
                              title = dict(
                                  text = 'Fecha')
                              ),
                          yaxis = dict(
                              showaxeslabels = False,
                              ticktext = y_labels,
                              tickvals = y_range,
                              title = dict(
                                  text = '')
                              ),
                          zaxis = dict(
                              ticks='outside',
                              ticksuffix='%',
                              title = dict(
                                  text = 'Population')
                              ),
                          ),
                      )
    fig.show(renderer="browser")
else:
    import matplotlib
    
    if use_inline:
        matplotlib.use('module://matplotlib_inline.backend_inline')
    else:
        matplotlib.use('Qt5Agg')

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Fecha')
    ax.set_xticklabels(x_labels)
    
    ax.set_ylabel('')
    ax.set_yticks(y_range)
    ax.set_yticklabels(y_labels)
    
    ax.set_zlabel('Population %')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, alpha=1)
    surf_aprox = ax.plot_surface(X, Y, Z_aprox, cmap=plt.cm.inferno, alpha=0.5)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
