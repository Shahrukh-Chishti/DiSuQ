import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
from IPython.display import Image, display
import time
from pyvis import network as pvnet
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from numpy import full,nan,zeros,arange

def empty(shape):
    E = zeros(shape)
    E.fill(nan)
    return E

# analysis plots
    
def render(fig,title,size,html,export):
    fig.update_layout(font={'size':30})
    fig.update_layout(margin_t=10,margin_r=10)
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="right",x=0.99))
    if size:
        height,width = size
        fig.update_layout(width=width,height=height)
    if export:
        fig.write_image('./'+title+'.'+export)
        time.sleep(2)
        fig.write_image('./'+title+'.'+export)
    fig.update_layout(title=title)
    fig.update_layout(margin_t=100)
    if html:    
        fig.write_html('./'+title+'.html')
    else :
        py.iplot(fig)
        
def plotBox(plot,title=None,x_label=None,y_label=None,size=None,html=False,export=None):
    fig = go.Figure()
    for name,dist in plot.items():
        fig.add_trace(go.Box(y=dist,name=name))
    fig.update_layout(xaxis_title=x_label,yaxis_title=y_label)
    fig.update_layout(showlegend=False)
    
    render(fig,title,size,html,export)        

def plotCombine(plot,title=None,x_label=None,y_label=None,mode='lines',size=None,html=False,export=None):
    fig = go.Figure()
    for name,(x,y) in plot.items():
        fig.add_trace(go.Scatter(x=x,y=y,name=name,mode=mode,line={'width':5}))
    fig.update_layout(xaxis_title=x_label,yaxis_title=y_label)
    
    render(fig,title,size,html,export)
    
def plotScatter(x,plot,title=None,x_label=None,y_label=None,size=None,dot_size=1,html=False,export=None,legend=True):
    fig = go.Figure()
    for name,values in plot.items():
        fig.add_trace(go.Scatter(x=x,y=values,name=name,mode='markers',marker={'size':dot_size}))
    fig.update_layout(xaxis_title=x_label,yaxis_title=y_label)
    fig.update_layout(showlegend=legend)
    render(fig,title,size,html,export)

def plotCompare(x,plot,title=None,x_label=None,y_label=None,size=None,html=False,export=None,legend=True):
    fig = go.Figure()
    for name,values in plot.items():
        fig.add_trace(go.Scatter(x=x,y=values,name=name,line={'width':5}))
    fig.update_layout(xaxis_title=x_label,yaxis_title=y_label)
    fig.update_layout(showlegend=legend)
    render(fig,title,size,html,export)

def plotHeatmap(z,x,y,title=None,xaxis=None,yaxis=None,size=None,html=False,export=None,ncontours=10,legend=True,log=False):
    fig = go.Figure()
    start=z.min(); end=z.max()
    contours = dict(start=start,end=end,size=(end-start)/ncontours)
    heatmap = go.Contour(z=z,y=y,x=x,contours_coloring='heatmap',contours=contours)
    fig.add_trace(heatmap)
    fig.update_layout(showlegend=legend)
    fig.update_traces(showscale=legend)
    fig.update_layout(coloraxis_showscale=legend)
    fig.update_layout(xaxis_title=xaxis,yaxis_title=yaxis)
    if log:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    
    render(fig,title,size,html,export)
    
def plotTrajectory(evo,plot,title=None,x_label=None,y_label=None,size=None,html=False,export=None,legend=False):
    fig = go.Figure()
    evo = evo/max(evo)*20
    for name,path in plot.items():
        fig.add_trace(go.Scatter(x=path[:,0],y=path[:,1],name=name,mode='lines+markers',
                                marker={'size':evo},line={'width':5}))
    fig.update_layout(xaxis_title=x_label,yaxis_title=y_label)
    fig.update_layout(showlegend=legend)
    render(fig,title,size,html,export)
    
def plotOptimization(z,x,y,paths,title=None,xaxis=None,yaxis=None,size=None,html=False,export=None,legend=False,log=False):
    fig = go.Figure()
    heatmap = go.Contour(z=z,y=y,x=x,name='loss',contours_coloring='heatmap')
    fig.add_trace(heatmap)
    fig.update_layout(xaxis_title=xaxis,yaxis_title=yaxis)
    for name,path in paths.items():
        evo = arange(len(path)) ; evo = evo/max(evo)*15
        fig.add_trace(go.Scatter(x=path[:,0],y=path[:,1],name=name,mode='lines+markers',marker={'size':evo},line={'width':3}))
    fig.update_layout(showlegend=legend)
    if log:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    render(fig,title,size,html,export)
    
# graph plots
    
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