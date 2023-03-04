import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
from IPython.display import Image, display
from pyvis import network as pvnet
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from numpy import full,nan,zeros

def empty(shape):
    E = zeros(shape)
    E.fill(nan)
    return E

def plotCombine(plot,title=None,x_label=None,y_label=None,mode='lines'):
    fig = go.Figure()
    for name,(x,y) in plot.items():
        fig.add_trace(go.Scatter(x=x,y=y,name=name,mode=mode))
    fig.update_layout(title=title,
    xaxis_title=x_label,
    yaxis_title=y_label)
    py.iplot(fig)

def plotCompare(x,plot,title=None,x_label=None,y_label=None,save=False):
    fig = go.Figure()
    for name,values in plot.items():
        fig.add_trace(go.Scatter(x=x,y=values,name=name))
    fig.update_layout(title=title,
    xaxis_title=x_label,
    yaxis_title=y_label)
    if save:
        fig.write_html('./'+title+'.html')
    py.iplot(fig)

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
    pdot.write_png(filename+'.png')
    #view_pydot(pdot)

def plotHeatmap(z,x,y,title=None,xaxis=None,yaxis=None,aspect=None):
    fig = go.Figure()
    heatmap = go.Heatmap(z=z,y=y,x=x)
    fig.add_trace(heatmap)
    fig.update_layout(title=title,xaxis_title=xaxis,yaxis_title=yaxis)
    if aspect:
        fig.update_layout(width=aspect,height=aspect)
        fig.update_layout(yaxis={'scaleanchor':'x','scaleratio':aspect})
    py.iplot(fig)
    
def plotTrajectory(evo,plot,title=None,x_label=None,y_label=None,save=False):
    fig = go.Figure()
    evo = evo/max(evo)*10
    for name,path in plot.items():
        fig.add_trace(go.Scatter(x=path[:,0],y=path[:,1],name=name,mode='lines+markers',
                                marker={'size':evo}))
    fig.update_layout(title=title,
    xaxis_title=x_label,
    yaxis_title=y_label)
    #fig.update_layout(hovermode='x unified')
    if save:
        fig.write_html('./'+title+'.html')
    py.iplot(fig)
    
def plotOptimization(z,x,y,paths,title=None,xaxis=None,yaxis=None,aspect=None):
    fig = go.Figure()
    heatmap = go.Heatmap(z=z,y=y,x=x)
    fig.add_trace(heatmap)
    fig.update_layout(title=title,xaxis_title=xaxis,yaxis_title=yaxis)
    for name,path in paths.items():
        fig.add_trace(go.Scatter(x=path[:,0],y=path[:,1],name=name,mode='lines+markers'))
    if aspect:
        fig.update_layout(width=aspect,height=aspect)
        fig.update_layout(yaxis={'scaleanchor':'x','scaleratio':aspect})
    py.iplot(fig)