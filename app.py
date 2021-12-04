from sklearn import datasets
from rich.console import RenderGroup
from rich.live import Live

import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd
from numpy import transpose, multiply, dot
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
from numpy.linalg import eig
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn import datasets

from datetime import datetime
import sys
from time import sleep

global prev_clicks, accuracy, svm1, svm2, svm3, c1_X_test, c2_X_test, c3_X_test, c1_Y_test, c2_Y_test, c3_Y_test
global prev_c1_button, prev_c2_button, prev_c3_button

prev_clicks = 0
prev_c1_button = 0
prev_c2_button = 0
prev_c3_button = 0
accuracy = 0
svm1,svm2,svm3 = None, None, None

def main():
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.Div([
        	html.Div(id="header", children=[
                html.H1('DASHBOARD', id="dashboard-title"),
                html.Div(id="dataset", children=[
                   html.H2("Dataset: ", id="dataset-text"),
                   dcc.Dropdown(id='dataset-dropdown',
                       options=[
                           {'label': 'SIMULATION', 'value': 'simulation'},
                           {'label': 'IRIS', 'value': 'iris'},
                       ],
                       value='simulation',
                       clearable = False,
                       searchable = False
                   ),
                ]),
                ]),
            dcc.Graph(id='graph'),
        	html.Div(children=[
                html.Div(id="hyperparameters", children=[
                    html.H2("Hyperparameters:"),
                    html.Div([
                        html.H4("Max_iterations: ", id='maxiteration-text'),
                        dcc.RadioItems(
                            id='maxiteration-options',
                            options=[
                                {'label': '10', 'value': 10},
                                {'label': '50', 'value': 50},
                                {'label': '100', 'value': 100},
                                {'label': '500', 'value': 500},
                                {'label': '1000', 'value': 1000},
                            ],
                            value=10
                        )
                    ]),
                    html.Div([
                        html.H4("Learning Rate:", id='learningrate-text'),
                        dcc.RadioItems(
                            id='learningrate-options',
                            options=[
                                {'label': '0.00001', 'value': 0.00001},
                                {'label': '0.0001', 'value': 0.0001},
                                {'label': '0.001', 'value': 0.001},
                                {'label': '0.01', 'value': 0.01},
                                {'label': '0.1', 'value': 0.1},
                                {'label': '0.5', 'value': 0.5},
                                {'label': '1', 'value': 1}
                            ],
                            value=1
                        )
                    ]),
                    html.Div([
                        html.H4("Regularization Parameter: ", id='regularization-text'),
                        dcc.RadioItems(
                            id='regularization-options',
                            options=[
                                {'label': '0.01', 'value': 0.01},
                                {'label': '0.05', 'value': 0.05},
                                {'label': '0.1', 'value': 0.1},
                                {'label': '0.25', 'value': 0.25},
                                {'label': '0.5', 'value': 0.5},
                                {'label': '1', 'value': 1}
                            ],
                            value=1
                        )
                    ]),
                ]),
                html.Div(id="buttons", children=[
                    html.Div(id='train', children=[
                        html.Button('Train Class 1', id='c1-button', n_clicks_timestamp=0),
                        html.Button('Train Class 2', id='c2-button', n_clicks_timestamp=0),
                        html.Button('Train Class 3', id='c3-button', n_clicks_timestamp=0),
                    ]),
                    html.Div(id='test', children=[
                        html.Button('Test!', id='test-button', n_clicks=0),
                    ]),
                    html.H3(id='accuracy')
                ]),
                html.Div(id="train-results", children=[
                    html.H2("Parameters: "),
                    html.H4(id='c1-parameters'),
                    html.H4(id='c2-parameters'),
                    html.H4(id='c3-parameters'),
                ]),
    	        html.Div(children=[
                    dcc.Graph(id='graph-c1'),
                    dcc.Graph(id='graph-c2'),
                    dcc.Graph(id='graph-c3')
                ])
            ])
        ])
    ])

    #	Callback for updating the svm graph
    @app.callback([Output('graph', 'figure'),
                  Output('graph-c1', 'figure'),
                  Output('graph-c2', 'figure'),
                  Output('graph-c3', 'figure'),
                  Output('c1-parameters', 'children'),
                  Output('c2-parameters', 'children'),
                  Output('c3-parameters', 'children'),
                  Output('accuracy', 'children')],
                    Input('dataset-dropdown', 'value'),
                    Input('c1-button', 'n_clicks_timestamp'),
                    Input('c2-button', 'n_clicks_timestamp'),
                    Input('c3-button', 'n_clicks_timestamp'),
                    Input('maxiteration-options', 'value'),
                    Input('learningrate-options', 'value'),
                    Input('regularization-options', 'value'),
                    Input('graph', 'figure'),
                    Input('graph-c1', 'figure'),
                    Input('graph-c2', 'figure'),
                    Input('graph-c3', 'figure'),
                    Input('c1-parameters', 'children'),
                    Input('c2-parameters', 'children'),
                    Input('c3-parameters', 'children'),
                    Input('test-button', 'n_clicks'))
    def update_graph(dataset, c1_button, c2_button, c3_button, max_iterations, learning_rate, regularization_parameter, graph, graph_c1, graph_c2, graph_c3,
                        c1_parameters, c2_parameters, c3_parameters, n_clicks):
        global prev_clicks, accuracy, svm1, svm2, svm3, c1_X_test, c2_X_test, c3_X_test, c1_Y_test, c2_Y_test, c3_Y_test
        global c1_X_train, c1_Y_train, c1_X_test, c1_Y_test, data_c1
        global c2_X_train, c2_Y_train, c2_X_test, c2_Y_test
        global c3_X_train, c3_Y_train, c3_X_test, c3_Y_test
        global data_c1, data_c2, data_c3, pca_data
        global prev_c1_button, prev_c2_button, prev_c3_button

        if (dataset == 'simulation'):
            data = get_simulation_dataset()
            colormap = {'Class 1':'mediumseagreen', 'Class 2':'red', 'Class 3':'dodgerblue'}
        elif (dataset == 'iris'):
            data = get_iris_dataset()
            colormap = {'Setosa':'mediumseagreen', 'Versicolour':'red', 'Virginica':'dodgerblue'}

        if (n_clicks > prev_clicks):
            if (svm1 != None and svm2 != None and svm3 != None):
                #test_X = pd.concat([pd.DataFrame(standardize(c1_X_test)), pd.DataFrame(standardize(c2_X_test)), pd.DataFrame(standardize(c3_X_test))], axis=0)
                #test_Y = pd.concat([pd.DataFrame(c1_Y_test), pd.DataFrame(c2_Y_test), pd.DataFrame(c3_Y_test)], axis=0)
                #test_Y = test_Y.rename(columns={0:'label'})
                #test_data = pd.concat([test_X, test_Y], axis=1)
                test_data = pca_data

                successes = 0
                for i in range(0, len(test_data)):
                    if (dataset == 'iris'):
                        prediction = predict_multiclass_iris(test_data.iloc[i].drop('label'),svm1,svm2,svm3)
                    else:
                        prediction = predict_multiclass_simulation(test_data.iloc[i].drop('label'),svm1,svm2,svm3)
                    if prediction == int(test_data['label'].iloc[i]):
                        successes +=1

                accuracy = (successes / len(test_data)) * 100

            return graph, graph_c1, graph_c2, graph_c3, c1_parameters, c2_parameters, c3_parameters, [html.H4(f"Accuracy: {'%.2f' % accuracy}%")]

        if (int(c1_button) == 0 and int(c2_button) == 0 and int(c3_button) == 0):
            if (dataset == 'iris'):
                pca_data = PCA.calculate(pd.DataFrame(data), 2)
                pca_data['label'] = data['label'].astype('category')
                pca_data['label'] = pca_data['label'].replace({1:'Setosa', 2:'Versicolour', 3:'Virginica'})
            else:
                pca_data = data
                pca_data['label'] = data['label'].astype('category')
                pca_data['label'] = pca_data['label'].replace({1:'Class 1', 2:'Class 2', 3:'Class 3'})


            graph = go.Figure(px.scatter(pca_data , x=0, y=1,
                                        symbol='label', color='label',
                                        labels={"0":"X Feature", "1":"Y Feature"},
                                        title=f"{dataset.upper()} DATASET",
                                        color_discrete_map=colormap))

            if (dataset == 'iris'):
                pca_data['label'] = pca_data['label'].replace({'Setosa':'1', 'Versicolour':'2', 'Virginica':'3'})
            else:
                pca_data['label'] = pca_data['label'].replace({'Class 1':1, 'Class 2':2, 'Class 3':3})

            if (dataset == 'simulation'):
                graph.update_traces(marker=dict(size=12))
            graph.update_layout(plot_bgcolor='#303030', paper_bgcolor='#303030')
            graph.update_layout(font_color='white')
            graph.update_layout(title={'x':0.5, 'xanchor': 'center'})
            graph.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
            graph.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
            graph.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
            graph.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')

            '''     CLASS 1          '''

            data_c1 = data[data.label != 3]
            data_c1['label'] = data_c1['label'].replace(to_replace=1, value=1).replace(to_replace=2, value=-1)
            data_c1_pca = PCA.calculate(data_c1.drop('label', axis=1), 2)
            data_c1_pca['label'] = data_c1['label']

            if (dataset == 'iris'):
                c1_X_train, c1_Y_train, c1_X_test, c1_Y_test = split_and_transform(data_c1_pca)
            else:
                c1_X_train, c1_Y_train, c1_X_test, c1_Y_test = split(data_c1)

            data_c1 = pd.DataFrame(c1_X_train)
            data_c1['label'] = pd.DataFrame(c1_Y_train)

            if (dataset == 'iris'):
                data_c1['label'] = data_c1['label'].astype('category')
                data_c1['label'] = data_c1['label'].replace({1:'Setosa', -1:'Versicolour'})
            else:
                data_c1['label'] = data_c1['label'].astype('category')
                data_c1['label'] = data_c1['label'].replace({1:'Class 1', -1:'Class 2'})

            graph_c1 = go.Figure(px.scatter(data_c1, x=0, y=1,
                                    symbol='label', color='label',
                                    labels={"0":"X Feature", "1":"Y Feature"},
                                    color_discrete_map=colormap))

            if (dataset == 'iris'):
                data_c1['label'] = data_c1['label'].replace({'Setosa':1, 'Versicolour':-1})
            else:
                data_c1['label'] = data_c1['label'].replace({'Class 1':1, 'Class 2':-1})



            '''     CLASS 2       '''
            data_c2 = data[data.label != 2]
            data_c2['label'] = data_c2['label'].replace(to_replace=1, value=1).replace(to_replace=3, value=-1)
            data_c2.dropna(how="all", inplace=True) # remove any empty lines
            data_c2.reset_index(inplace=True)
            data_c2.drop(['index'], axis=1, inplace=True)
            data_c2_pca = PCA.calculate(data_c2.drop(['label'], axis=1), 2)
            data_c2_pca['label'] = data_c2['label']

            if (dataset == 'iris'):
                c2_X_train, c2_Y_train, c2_X_test, c2_Y_test = split_and_transform(data_c2_pca)
            else:
                c2_X_train, c2_Y_train, c2_X_test, c2_Y_test = split(data_c2)

            data_c2 = pd.DataFrame(c2_X_train)
            data_c2['label'] = pd.DataFrame(c2_Y_train)

            if (dataset == 'iris'):
                data_c2['label'] = data_c2['label'].astype('category')
                data_c2['label'] = data_c2['label'].replace({1:'Setosa', -1:'Virginica'})
            else:
                data_c2['label'] = data_c2['label'].astype('category')
                data_c2['label'] = data_c2['label'].replace({1:'Class 1', -1:'Class 3'})

            graph_c2 = go.Figure(px.scatter(data_c2, x=0, y=1,
                                    symbol='label', color='label',
                                    labels={"0":"X Feature", "1":"Y Feature"},
                                    color_discrete_map=colormap))

            if (dataset == 'iris'):
                data_c2['label'] = data_c2['label'].replace({'Setosa':1, 'Virginica':-1})
            else:
                data_c2['label'] = data_c2['label'].replace({'Class 1':1, 'Class 3':-1})

            '''      CLASS 3     '''
            data_c3 = data[data.label != 1]
            data_c3['label'] = data_c3['label'].replace(to_replace=2, value=1).replace(to_replace=3, value=-1)
            data_c3.dropna(how="all", inplace=True) # remove any empty lines
            data_c3.reset_index(inplace=True)
            data_c3.drop(['index'], axis=1, inplace=True)
            data_c3_pca = PCA.calculate(data_c3.drop(['label'], axis=1), 2)
            data_c3_pca['label'] = data_c3['label']

            if (dataset == 'iris'):
                c3_X_train, c3_Y_train, c3_X_test, c3_Y_test = split_and_transform(data_c3_pca)
            else:
                c3_X_train, c3_Y_train, c3_X_test, c3_Y_test = split(data_c3)

            data_c3 = pd.DataFrame(c3_X_train)
            data_c3['label'] = pd.DataFrame(c3_Y_train)

            if (dataset == 'iris'):
                data_c3['label'] = data_c3['label'].astype('category')
                data_c3['label'] = data_c3['label'].replace({1:'Versicolour', -1:'Virginica'})
            else:
                data_c3['label'] = data_c3['label'].astype('category')
                data_c3['label'] = data_c3['label'].replace({1:'Class 2', -1:'Class 3'})

            graph_c3 = go.Figure(px.scatter(data_c3, x=0, y=1,
                                    symbol='label', color='label',
                                    labels={"0":"X Feature", "1":"Y Feature"},
                                    color_discrete_map=colormap))
            if (dataset == 'iris'):
                data_c3['label'] = data_c3['label'].replace({'Versicolour':1, 'Virginica':-1})
            else:
                data_c3['label'] = data_c3['label'].replace({'Class 2':1, 'Class 3':-1})

            train_figs = [graph_c1, graph_c2, graph_c3]

            '''        UPDATE FIGS       '''

            for figure in train_figs:
                figure.update_layout(plot_bgcolor='#303030', paper_bgcolor='#303030')
                figure.update_coloraxes(showscale=False)
                #figure.update_layout(showlegend=False)
                figure.update_layout(font_color='white')
                figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                figure.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
                figure.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
                figure.update_layout(legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.9
                ))

            return graph, graph_c1, graph_c2, graph_c3, [html.H4(f"Class 1 - W: {None} b: {None}")], [html.H4(f"Class 2 - W: {None} b: {None}")], [html.H4(f"Class 3 - W: {None} b: {None}")], [html.H3(f"Accuracy: {'%.2f' % accuracy}%")]

        elif (int(c1_button) > int(c2_button) and int(c1_button) > int(c3_button) and int(c1_button) > prev_c1_button):
            prev_c1_button = int(c1_button)

            graph_c1 = go.Figure(graph_c1)
            graph_c2 = go.Figure(graph_c2)
            graph_c3 = go.Figure(graph_c3)

            svm1 = SVM(learning_rate, regularization_parameter, max_iterations)

            svm1.fit(c1_X_train, c1_Y_train)

            W1 = svm1.W
            b1 = svm1.b
            acc = svm1.accuracy(c1_X_test, c1_Y_test)

            dimensions = len(data_c1.columns) - 1

            w_norm = np.sqrt(dot(svm1.W, svm1.W))
            margin = 1 / w_norm

            min_x = min(data_c1[0])
            max_x = max(data_c1[1])
            x = np.linspace(min_x,max_x,2)
            m = -svm1.W[0] / svm1.W[1]
            y = m*x + svm1.b/svm1.W[1]
            y_lower = m*x + (svm1.b - margin)/svm1.W[1]
            y_upper = m*x + (svm1.b + margin)/svm1.W[1]

            avg_x = (max(data_c1[0]) + min(data_c1[0])) / 2
            avg_y = (max(data_c1[1]) + min(data_c1[1])) / 2

            graph_c1.add_trace(go.Scatter(x=x, y=y))
            graph_c1.data[-1].name = 'Hyperplane'
            graph_c1.add_trace(go.Scatter(x=x, y=y_lower, line=dict(dash='dash')))
            graph_c1.data[-1].name = 'Margin L'
            graph_c1.add_trace(go.Scatter(x=x, y=y_upper, line=dict(dash='dash')))
            graph_c1.data[-1].name = 'Margin H'
            graph_c1.update_xaxes(range=[min(data_c1[0]) / 1.1, max(data_c1[0]) * 1.1])
            graph_c1.update_yaxes(range=[min(data_c1[1]) / 1.1, max(data_c1[1]) * 1.1])

            return graph, graph_c1, graph_c2, graph_c3, [html.H4(f'Class 1 - W: [{"%.2f" % W1[0]},{"%.2f" % W1[1]}] b: {"%.2f" % b1} Acc: {"%.2f" % acc}%')], c2_parameters, c3_parameters, [html.H3(f"Accuracy: {'%.2f' % accuracy}%")]

        elif (int(c2_button) > int(c1_button) and int(c2_button) > int(c3_button) and int(c2_button) > prev_c2_button):
            prev_c2_button = int(c2_button)

            graph_c1 = go.Figure(graph_c1)
            graph_c2 = go.Figure(graph_c2)
            graph_c3 = go.Figure(graph_c3)


            svm2 = SVM(learning_rate, regularization_parameter, max_iterations)

            svm2.fit(c2_X_train, c2_Y_train)

            W2 = svm2.W
            b2 = svm2.b
            acc = svm2.accuracy(c2_X_test, c2_Y_test)

            dimensions = len(data_c2.columns) - 1

            w_norm = np.sqrt(dot(svm2.W, svm2.W))
            margin = 1 / w_norm

            min_x = min(data_c2[0])
            max_x = max(data_c2[1])
            x = np.linspace(min_x,max_x,2)
            m = -svm2.W[0] / svm2.W[1]
            y = m*x + svm2.b/svm2.W[1]
            y_lower = m*x + (svm2.b - margin)/svm2.W[1]
            y_upper = m*x + (svm2.b + margin)/svm2.W[1]

            avg_x = (max(data_c2[0]) + min(data_c2[0])) / 2
            avg_y = (max(data_c2[1]) + min(data_c2[1])) / 2

            graph_c2.add_trace(go.Scatter(x=x, y=y))
            graph_c2.data[-1].name = 'Hyperplane'
            graph_c2.add_trace(go.Scatter(x=x, y=y_lower, line=dict(dash='dash')))
            graph_c2.data[-1].name = 'Margin L'
            graph_c2.add_trace(go.Scatter(x=x, y=y_upper, line=dict(dash='dash')))
            graph_c2.data[-1].name = 'Margin H'
            graph_c2.update_xaxes(range=[min(data_c2[0]) / 1.1, max(data_c2[0]) * 1.1])
            graph_c2.update_yaxes(range=[min(data_c2[1]) / 1.1, max(data_c2[1]) * 1.1])

            return graph, graph_c1, graph_c2, graph_c3, c1_parameters, [html.H4(f'Class 2 - W: [{"%.2f" % W2[0]},{"%.2f" % W2[1]}] b: {"%.2f" % b2} Acc: {"%.2f" % acc}%')], c3_parameters, [html.H3(f"Accuracy: {'%.2f' % accuracy}%")]

        elif (int(c3_button) > int(c1_button) and int(c3_button) > int(c2_button) and int(c3_button) > prev_c3_button):
            prev_c3_button = int(c3_button)

            graph_c1 = go.Figure(graph_c1)
            graph_c2 = go.Figure(graph_c2)
            graph_c3 = go.Figure(graph_c3)

            svm3 = SVM(learning_rate, regularization_parameter, max_iterations)

            svm3.fit(c3_X_train, c3_Y_train)

            W3 = svm3.W
            b3 = svm3.b
            acc = svm3.accuracy(c3_X_test, c3_Y_test)


            dimensions = len(data_c3.columns) - 1

            w_norm = np.sqrt(dot(svm3.W, svm3.W))
            margin = 1 / w_norm

            min_x = min(data_c3[0])
            max_x = max(data_c3[1])
            x = np.linspace(min_x,max_x,2)
            m = -svm3.W[0] / svm3.W[1]
            y = m*x + svm3.b/svm3.W[1]
            y_lower = m*x + (svm3.b - margin)/svm3.W[1]
            y_upper = m*x + (svm3.b + margin)/svm3.W[1]

            avg_x = (max(data_c3[0]) + min(data_c3[0])) / 2
            avg_y = (max(data_c3[1]) + min(data_c3[1])) / 2

            graph_c3.add_trace(go.Scatter(x=x, y=y))
            graph_c3.data[-1].name = 'Hyperplane'
            graph_c3.add_trace(go.Scatter(x=x, y=y_lower, line=dict(dash='dash')))
            graph_c3.data[-1].name = 'Margin L'
            graph_c3.add_trace(go.Scatter(x=x, y=y_upper, line=dict(dash='dash')))
            graph_c3.data[-1].name = 'Margn H'
            graph_c3.update_xaxes(range=[min(data_c3[0]) / 1.1, max(data_c3[0]) * 1.1])
            graph_c3.update_yaxes(range=[min(data_c3[1]) / 1.1, max(data_c3[1]) * 1.1])

            return graph, graph_c1, graph_c2, graph_c3, c1_parameters, c2_parameters, [html.H4(f'Class 3 - W: [{"%.2f" % W3[0]},{"%.2f" % W3[1]}] b: {"%.2f" % b3} Acc: {"%.2f" % acc}%')], [html.H3(f"Accuracy: {'%.2f' % accuracy}%")]

        return graph, graph_c1, graph_c2, graph_c3, c1_parameters, c2_parameters, c3_parameters, [html.H3(f"Accuracy: {'%.2f' % accuracy}%")]

    app.run_server(debug=True)

class SVM:
    # Initialize our SVM with our hyperparameters.
    def __init__(self, learning_rate, regularization_parameter, max_iterations):
        self.learning_rate = learning_rate
        self.C = regularization_parameter
        self.max_iterations = max_iterations
        self.W = None
        self.b = None

    # Train our SVM using stochastic gradient descent
    def fit(self, X, Y):
        n_features = len(X[0])
        n_samples = len(X)

        # Initialize the weights/hyperparameters of SVM
        self.W = np.array([0.0] * n_features)
        self.b = 0

        # Initialize the values for the subgradients of W and b
        w_grad = np.array([0.0] * n_features)
        b_grad = 0

        hinge_prev = self.hinge_loss(X, Y)

        # Perform SGD to update our parameters for our specified number of iterations
        for i in range(0, self.max_iterations):
            for x_i, y_i in zip(X,Y):
                fx_i = np.dot(self.W, x_i) + self.b
                t = y_i * fx_i

                if (t < 1):
                    w_grad += -1 * (y_i * x_i)
                    b_grad += -1 * y_i
                else:
                    continue

            w_grad = self.W + (self.C * w_grad)
            b_grad = self.C * b_grad

            self.W -= self.learning_rate * w_grad
            self.b -= self.learning_rate * b_grad

            hinge_cur = self.hinge_loss(X, Y)

            if (hinge_cur > hinge_prev and (n_features == 2 or n_features == 3)):
                break

            hinge_prev = hinge_cur

            print("Svm is fit.")

        return

    # Predicts binary classification of a given input
    def predict(self, x):
        return 1 if dot(self.W, transpose(x)) + self.b >= 0 else -1

    # Prints the accuracy of the model using the current weights
    def accuracy(self, X, Y):
        print("Testing accuracy...")
        print()
        successes = 0
        n = len(X)

        for i in range(0, n):
            if (self.predict(X[i]) == Y[i]):
                successes += 1
        return successes / n * 100

    # Calculates the hinge loss of the given data using the current weights
    def hinge_loss(self, X, Y):
        distance_sum = 0

        for x_i, y_i in zip(X,Y):
            distance_sum += max(0, 1 - y_i * (dot(self.W, x_i) + self.b))

        regularizer = 0.5 * dot(transpose(self.W), self.W)
        error_term = self.C * distance_sum

        loss = regularizer + error_term

        return float(loss)

class PCA:
    def calculate(X, dimensions):
        # Center the data about the origin and standardize it
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        std_data = (X - X_mean) / X_std

        # Calculate the covariance matrix and the eigenspace
        covariance = std_data.cov()
        values, vectors = eig(covariance)

        # Sort the eigenvectors in order of largest eigenvalue
        sorted_index = np.argsort(values)[::-1]
        values = values[sorted_index]
        vectors = vectors[sorted_index]

        # Get top-most eigenvectors of specified dimensionality
        components = vectors[0:dimensions]

        # Undo the standardization
        unstd_data = (std_data * X_std) + X_mean

        # Project the dataset and obtain the coordinates
        coordinates = dot(components, transpose(unstd_data))

        return pd.DataFrame(transpose(coordinates))

def standardize(X):
    return (X - X.mean()) / X.std()

def split_and_transform(data):
    # Split data into train and test sets
    train, test = train_test_split(data, test_size=0.15)

    X_train = np.array(standardize(train.drop(columns=['label'])))
    Y_train = np.array(train['label'])
    X_test = np.array(standardize(test.drop(columns=['label'])))
    Y_test = np.array(test['label'])

    return X_train, Y_train, X_test, Y_test

def split(data):
    # Split data into train and test sets
    train, test = train_test_split(data, test_size=0.15)

    X_train = np.array(train.drop(columns=['label']))
    Y_train = np.array(train['label'])
    X_test = np.array(test.drop(columns=['label']))
    Y_test = np.array(test['label'])

    return X_train, Y_train, X_test, Y_test

def predict_multiclass_simulation(data, svm1, svm2, svm3):
    val1 = svm1.predict(data)
    val2 = svm2.predict(data)
    val3 = svm3.predict(data)

    if (val1 == 1 and val2 == 1):
        return 1
    elif (val2 == -1 and val3 == -1):
        return 3
    else:
        return 2

def predict_multiclass_iris(data, svm1, svm2, svm3):
    val1 = svm1.predict(data)
    val2 = svm2.predict(data)
    val3 = svm3.predict(data)

    if (val1 == 1 and val2 == -1):
        return 1
    elif (val3 == -1 and val1 == 1):
        return 2
    else:
        return 3


def get_simulation_dataset():
    X = pd.DataFrame([[-2, 2], [-3, 2], [-2, -2], [-3, -2.5], [-2, 3], [-1, -3], [-1, 1], [-1, 3], [-3, 0], [-2, -1], [-1, -2], [1, 3], [1, 2], [0,3], [0, 2], [1, 4], [3, 2.5], [2, 2.5], [2, 1], [3, .5], [0, 4], [2, 4], [0,-1], [1, -1], [1, -3], [2, 0], [2, -3], [3, -3], [2, -2], [0, -2], [0, 1], [2.5, 0], [3, -1.5]])
    Y = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]).rename(columns={0:'label'})
    return pd.concat([X, Y], axis=1)


def get_iris_dataset():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data)
    iris_df['label'] = iris.target
    iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'label']
    iris_df.dropna(how="all", inplace=True) # remove any empty lines
    iris_df.label.replace({0:1, 1:2, 2:3}, inplace=True)

    return iris_df

if __name__ == '__main__':
	main()
