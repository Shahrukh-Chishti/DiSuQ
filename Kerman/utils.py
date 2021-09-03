import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
from IPython.display import Image, display
from pyvis import network as pvnet
import networkx as nx
from networkx.drawing.nx_pydot import write_dot

def plotCompare(x,plot):
    fig = []
    for name,values in plot.items():
        fig.append(go.Scatter(x=x,y=values,name=name))
    py.plot(fig)

def view_pydot(G):
    #import ipdb; ipdb.set_trace()
    pdot = to_pydot(G)
    img = Image(pdot.create_png())
    display(img)

def plotVisGraph(G,filename='temp', height='300px', width='500px'):
    G = G.copy()
    net = pvnet.Network(height=height, width=width)
    net.from_nx(G)
    return net.show(filename+'.html')

def plotMatPlotGraph(G,filename):
    nx.draw(G,with_labels=True,pos=nx.circular_layout(G))
    plt.show()

def plotDOTGraph(G,filename='temp'):
    pdot = to_pydot(G)
    #DOT = nx.nx_agraph.write_dot(G,filename+'.png')
    import ipdb; ipdb.set_trace()
    pdot.write_png(filename+'.png')
    #view_pydot(pdot)
