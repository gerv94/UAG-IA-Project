# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys

# Read data from a csv
data = pd.read_csv('DatosOrdenados.csv', header=0, index_col=0)

z = np.array(data.values)
x_size, y_size = z.shape

x = np.arange(x_size)
y = np.arange(y_size)

X, Y = np.meshgrid(x, y)
Z = z[X,Y]

x_labels, y_labels = data.axes

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

fig.update_layout(title = 'Ocupation',
                  autosize = True,
                  margin = dict(l=65, r=50, b=65, t=90),
                  scene = dict(
                      xaxis = dict(
                          showticklabels = False,
                          ticktext = x_labels,
                          tickvals = x,
                          title = dict(
                              text = 'Fecha')
                          ),
                      yaxis = dict(
                          ticktext = y_labels,
                          tickvals = y,
                          title = dict(
                              text = 'Clase')
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

# For 3D subplots see: https://plotly.com/python/3d-subplots/