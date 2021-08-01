import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
from IPython.display import Image, display

def plotCompare(x,plot):
    fig = []
    for name,values in plot.items():
        fig.append(go.Scatter(x=x,y=values,name=name))
    py.plot(fig)

def view_pydot(pdot):
    #import ipdb; ipdb.set_trace()
    img = Image(pdot.create_png())
    display(img)

def plotDOTGraph(G,filename='temp'):
    pdot = to_pydot(G)
    pdot.write_png(filename+'.png')
    #view_pydot(pdot)
