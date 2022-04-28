import plotly.graph_objs as go
import plotly.offline as py

def plotCompare(x,plot,title=None,x_label=None,y_label=None):
    fig = go.Figure()
    for name,values in plot.items():
        fig.add_trace(go.Scatter(x=x,y=values,name=name))
    fig.update_layout(title=title,
    xaxis_title=x_label,
    yaxis_title=y_label)
    py.iplot(fig)
    
def plotHeatmap(z,x,y,title=None,xaxis=None,yaxis=None):
    fig = go.Figure()
    heatmap = go.Heatmap(z=z,y=y,x=x)
    fig.add_trace(heatmap)
    fig.update_layout(title=title,xaxis_title=xaxis,yaxis_title=yaxis)
    py.iplot(fig)