#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json

# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

YEAR=[2010, 2019]
ACCOUNT="A0001"

##############################################################################################################################################################
def network_graph(yearRange, AccountToSearch):

    edge1 = pd.read_csv('wetgrass_edge.csv')
    node1 = pd.read_csv('wetgrass_node.csv')


    accountSet=set() # contain unique account
    for index in range(0,len(edge1)):
        accountSet.add(edge1['Source'][index])
        accountSet.add(edge1['Target'][index])



    G = nx.from_pandas_edgelist(edge1, 'Source', 'Target', ['Source', 'Target', 'TransactionAmt', 'Date'], create_using=nx.MultiDiGraph())
    nx.set_node_attributes(G, node1.set_index('Account')['NodeName'].to_dict(), 'NodeName')
    nx.set_node_attributes(G, node1.set_index('Account')['Type'].to_dict(), 'Type')
    nx.set_node_attributes(G, node1.set_index('Account')['x'].to_dict(), 'x')
    nx.set_node_attributes(G, node1.set_index('Account')['y'].to_dict(), 'y')


    # pos = nx.layout.spring_layout(G)
    # pos = nx.layout.circular_layout(G)
    # nx.layout.shell_layout only works for more than 3 nodes


    for node in G.nodes:

        coords = [G.nodes[node]['x'],G.nodes[node]['y']]
        print(coords)
        G.nodes[node]['pos'] = coords




    traceRecode = []  # contains edge_trace, node_trace, middle_node_trace
    ############################################################################################################################################################
    colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
    colors = ['rgb' + str(x.rgb) for x in colors]

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        weight = float(G.edges[edge]['TransactionAmt']) / max(edge1['TransactionAmt']) * 10
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                           mode='lines',
                           line={'width': weight},
                           marker=dict(color=colors[index]),
                           line_shape='spline',
                           opacity=1)
        traceRecode.append(trace)
        index = index + 1
    ###############################################################################################################################################################
    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 50, 'color': 'LightSkyBlue'})

    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        print(tuple([x]))



        hovertext ="x"+str(G.nodes[node]['x'])+ "y"+str(G.nodes[node]['y'])+ "NodeName: " + str(G.nodes[node]['NodeName']) + "<br>" + "AccountType: " + str(
            G.nodes[node]['Type'])
        text = node1['Account'][index]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])
        index = index + 1

    traceRecode.append(node_trace)
    ################################################################################################################################################################
    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'},
                                    opacity=0)

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        hovertext = "From: " + str(G.edges[edge]['Source']) + "<br>" + "To: " + str(
            G.edges[edge]['Target']) + "<br>" + "TransactionAmt: " + str(
            G.edges[edge]['TransactionAmt']) + "<br>" + "TransactionDate: " + str(G.edges[edge]['Date'])
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(middle_hover_trace)
    #################################################################################################################################################################
    figure = {
        "data": traceRecode,
        "layout": go.Layout(title='Visualisation Interactive', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=600,
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                    ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                    x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                    y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=3,
                                    arrowsize=4,
                                    arrowwidth=1,
                                    opacity=1
                                ) for edge in G.edges]
                            )}
    return figure








######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    #########################Title
    html.Div([html.H1("Bayesian Network Graph")],
             className="row",
             style={'textAlign': "center"}),
    #############################################################################################define the row
    html.Div(
        className="row",
        children=[
            ##############################################left side two input components
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                            **Time Range To Visualize**
                            Slide the bar to define year range.
                            """)),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Slider(
                                id='my-range-slider',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.5,
                                marks={
                                    0.1: {'label': '0.1'},
                                    0.2: {'label': '0.2'},
                                    0.3: {'label': '0.3'},
                                    0.4: {'label': '0.4'},
                                    0.5: {'label': '0.5'},
                                    0.6: {'label': '0.6'},
                                    0.7: {'label': '0.7'},
                                    0.8: {'label': '0.8'},
                                    0.9: {'label': '0.9'},
                                    1: {'label': '1'}}
                            ),
                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
                        style={'height': '300px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            **Account To Search**
                            Input the account to visualize.
                            """)),
                            dcc.Input(id="input1", type="text", placeholder="Account"),
                            html.Div(id="output")
                        ],
                        style={'height': '300px'}
                    )
                ]
            ),

            ############################################middle graph component
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=network_graph(YEAR, ACCOUNT))],
            ),

            #########################################right side two output component
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Hover Data**
                            Mouse over values in the graph.
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px'}),

                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Click Data**
                            Click on points in the graph.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px'})
                ]
            )
        ]
    )
])

###################################callback for left side components
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value,input1):
    YEAR = value
    ACCOUNT = input1
    return network_graph(value, input1)
    # to update the global variable of YEAR and ACCOUNT
################################callback for right side components
@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('my-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [dash.dependencies.Input('my-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)



if __name__ == '__main__':
    app.run_server(debug=True)