# Homework 7 - Exercise 2
# Libraries
import pandas as pd
import plotly
import plotly.graph_objs as go

# Program
# Data Consolidation
data = pd.read_csv("vehicles.csv")

# Marker Properties
markersize = data['qsec']
markercolor = data['mpg']
markershape = data['am'].replace(0, "square").replace(1, "circle")

# Plotly Figure
fig1 = go.Scatter3d(x=data['wt'],
                    y=data['disp'],
                    z=data['hp'],
                    marker=dict(size=markersize,
                                color=markercolor,
                                symbol=markershape,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

# Plotly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="wt"),
                                yaxis=dict( title="disp"),
                                zaxis=dict(title="hp")),)

# Plot and save as HTML File
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("6DPlot.html"))
