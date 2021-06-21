import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

def plotCompare(x,plot):
    fig = []
    for name,values in plot.items():
        fig.append(go.Scatter(x=x,y=values,name=name))
    py.plot(fig)
