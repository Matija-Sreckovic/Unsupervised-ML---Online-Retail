import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table

def find_tertiles(table, col):
    min_value = min(table[col])
    max_value = max(table[col])
    first_tertile_count = 0
    second_tertile_count = 0
    found_first_tertile = found_second_tertile = False
    first_tertile_number = 0
    second_tertile_number = 0
    values = table[col].unique()
    values.sort()
    
    for i in values:
        rows_with_value_i = table.loc[table[col] == i]
        if not found_first_tertile:
            first_tertile_count += len(rows_with_value_i)
            if first_tertile_count > len(table)/3:
                found_first_tertile = True
                first_tertile_number = i
        if not found_second_tertile:
            second_tertile_count += len(rows_with_value_i)
            if second_tertile_count > (2*len(table))/3:
                found_second_tertile = True
                second_tertile_number = i
    
    return first_tertile_number, second_tertile_number


def standardize(column):
    min_value = min(column)
    max_value = max(column)
    column -= min_value
    column /= (max_value - min_value)
    return column


def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans


app = dash.Dash(__name__)

#--------------------------------------------------------------------
#Import and clean data (importing xlsx into pandas)
df = pd.read_excel("Online Retail.xlsx")

df.reset_index(inplace=True)
print(df[:5])

#--------------------------------------------------------------------
#App layout
app.layout = html.Div([
    html.H1("Online Retail Dataset - Dashboard", style={'text-align': 'center'}),

    dcc.Dropdown(id='n_clusters',
                 options=[
                     {"label":"2", "value":2},
                     {"label":"3", "value":3},
                     {"label":"4", "value":4},
                     {"label":"5", "value":5},
                     {"label":"6", "value":6},
                     {"label":"7", "value":7},
                     {"label":"8", "value":8},
                     {"label":"9", "value":9},
                     {"label":"10", "value":10}],
                     multi=False,
                     value=3,
                     style={'width':"40%"}
                 ),

    html.Button('Display Online Retail Dataset', id='display-retail', n_clicks=0),

    dash_table.DataTable(id='Online-Retail-Dataset',
                        style_table={'maxWidth': '600px', 'maxHeight': '400px', 'overflowY': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'}),

    html.H3("RFM-Cluster Relationship"),

    dash_table.DataTable(id='RFM-Cluster Relationship',
                         style_table={'maxWidth': '600px', 'maxHeight': '400px', 'overflowY': 'auto'},
                         style_cell={'textAlign': 'left', 'padding': '5px'}),

    html.Div([
        html.Div([
            html.Div([
                html.H4("Revenue per Cluster"),

                # First DataTable
                dash_table.DataTable(
                    id='revenue-per-cluster',
                    style_table={'width': '45%', 'marginRight': '5px'},  # Adjust width and spacing
                    style_cell={'textAlign': 'left'}
                )
            ]),

            html.Div([
                html.H4("Cluster Names (up to You!)"),

                # Second DataTable
                dash_table.DataTable(
                    id='cluster-names',
                    style_table={'width': '45%'},  # Adjust width
                    style_cell={'textAlign': 'left'}
                ),
            ])
        ], style={'display': 'flex', 'justifyContent': 'flex-start', 'alignItems': 'flex-start'})
    ]),

    html.H3("3D Scatterplot", style={'text-align': 'center'}),

    dcc.Graph(id='3D-plot', figure={}),

    html.H3("2D Scatterplot", style={'text-align': 'center'}),
    
    dcc.Graph(id='2D-plot', figure={}),
])

#--------------------------------------------------------------------
@app.callback(
    [Output(component_id='Online-Retail-Dataset', component_property='data'),
     Output(component_id='Online-Retail-Dataset', component_property='columns')],
    [Input(component_id='display-retail', component_property='n_clicks')],
)
def display_retail(n_clicks):
    columns1 = []
    data1 = []
    if n_clicks and n_clicks > 0:
            data1 = df.copy()
            columns1 = [{'name': col, 'id': col} for col in data1.columns]
            data1 = data1.to_dict('records')
    return data1, columns1


@app.callback(
    [Output(component_id='RFM-Cluster Relationship', component_property='data'),
     Output(component_id='RFM-Cluster Relationship', component_property='columns'),
     Output(component_id='revenue-per-cluster', component_property='data'),
     Output(component_id='revenue-per-cluster', component_property='columns'),
     Output(component_id='cluster-names', component_property='data'),
     Output(component_id='cluster-names', component_property='columns'),
     Output(component_id='3D-plot', component_property='figure'),
     Output(component_id='2D-plot', component_property='figure')],
     [Input(component_id='n_clusters', component_property='value')],
)
def update_dashboard(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The number of clusters chosen by user was: {}".format(option_slctd)

    retail = df.copy()

    # Convert InvoiceDate to datetime
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])    

    #Add total price of purchase as a column
    retail['TotalPrice'] = retail['Quantity']*retail['UnitPrice']

    #Convert CustomerID to int
    retail.loc[~pd.isnull(retail['CustomerID']), 'CustomerID'] = retail.loc[~pd.isnull(retail['CustomerID']), 'CustomerID'].astype(int)

    #Get the date of each customer's most recent purchase
    last_purchases = retail.groupby('CustomerID').apply(lambda df: df.loc[df.InvoiceDate.idxmax()])
    last_purchases = last_purchases[['CustomerID', 'InvoiceDate']]

    #Divide the interval before the earliest and latest most recent purchase into 3 intervals. It turns out the one third is around 124 days.
    first_day = min(last_purchases['InvoiceDate'])
    end_of_first_interval = first_day + timedelta(days = 124)
    end_of_second_interval = end_of_first_interval + timedelta(days = 124)
    end_of_third_interval = max(last_purchases['InvoiceDate'])
    last_purchases['RecencyScore'] = 1
    last_purchases.loc[(last_purchases['InvoiceDate'] > end_of_first_interval) & (last_purchases['InvoiceDate'] <= end_of_second_interval), 'RecencyScore'] = 2
    last_purchases.loc[(last_purchases['InvoiceDate'] > end_of_second_interval) & (last_purchases['InvoiceDate'] <= end_of_third_interval), 'RecencyScore'] = 3

    order_count = retail.groupby('CustomerID').CustomerID.count()
    order_count = pd.DataFrame(order_count)
    order_count.columns = ['NumberOrders']

    first_tertile_number, second_tertile_number = find_tertiles(order_count, 'NumberOrders')
    first_tertile_number, second_tertile_number

    rf_score = last_purchases
    rf_score.rename(columns={'InvoiceDate': 'LastPurchaseDate'}, inplace=True)
    rf_score['NumberPurchases'] = order_count['NumberOrders']

    #We now have Recency and Frequency.
    rf_score['FrequencyScore'] = 1
    rf_score.loc[rf_score['NumberPurchases'] > first_tertile_number, 'FrequencyScore'] = 2
    rf_score.loc[rf_score['NumberPurchases'] > second_tertile_number, 'FrequencyScore'] = 3

    #We now want to get the M score - M stands for 'Monetary Value'. We compute how much each customer spent in total.
    money_spent = retail.groupby('CustomerID').TotalPrice.sum()

    #We get the first and second tertile for monetary value...
    rfm_score = rf_score
    rfm_score['MoneySpent'] = money_spent
    first_tertile_monetary, second_tertile_monetary = find_tertiles(rfm_score, 'MoneySpent')

    #... and assign the M scores appropriately.
    rfm_score['MonetaryValueScore'] = 1
    rfm_score.loc[rfm_score['MoneySpent'] > first_tertile_monetary, 'MonetaryValueScore'] = 2
    rfm_score.loc[rfm_score['MoneySpent'] > second_tertile_monetary, 'MonetaryValueScore'] = 3

    #We isolate the R, F and M scores into a new DataFrame.
    rfm_df = rfm_score[['RecencyScore', 'FrequencyScore', 'MonetaryValueScore']]

    #Perform the clustering
    kmeans = perform_clustering(rfm_df, option_slctd)

    # Generate all possible combinations of 1, 2, 3 in 3 columns
    combinations = pd.DataFrame([(x, y, z) for x in [1, 2, 3] for y in [1, 2, 3] for z in [1, 2, 3]],
                                columns=['col1', 'col2', 'col3'])
    combinations.columns = ['RecencyScore', 'FrequencyScore', 'MonetaryValueScore']

    # Predict the cluster for each combination
    combinations['Cluster'] = kmeans.predict(combinations)
    
    columns1 = [{'name': col, 'id': col} for col in combinations.columns]
    data1 = combinations.to_dict('records')

    #We take the labels given by KMeans and turn them into a Series, that we then join as a column to rfm_score
    labelSeries = pd.Series(kmeans.labels_, name="Cluster")
    rfm_score.reset_index(drop=True, inplace=True)
    rfm_score_with_labels = pd.concat([rfm_score, labelSeries], axis=1)

    #Revenue by segment is then obtained by a simple groupby 
    cluster_names = pd.Series([i for i in range(option_slctd)], name="Cluster")
    revenue_by_segment = rfm_score_with_labels.groupby(["Cluster"]).MoneySpent.sum()
    revenue_by_segment = pd.concat([cluster_names, revenue_by_segment], axis=1)
    data2 = revenue_by_segment.to_dict('records')
    columns2 = [{'name': col, 'id': col} for col in revenue_by_segment.columns]

    cluster_descriptions = pd.DataFrame({
    "Description": ["" for i in range(option_slctd)]  # Editable column (initially empty)
    })
    cluster_descriptions = pd.concat([cluster_names, cluster_descriptions], axis=1)
    data3 = cluster_descriptions.to_dict('records')
    columns3 = [{"name": col, "id": col, "editable": True} for col in cluster_descriptions.columns]

    #Now we focus on getting a 3D scatterplot of RFM scores and a 2D scatterplot after applying PCA

    #Get the earliest purchase date
    first_purchase = min(retail['InvoiceDate'])
    rfm_score['FirstPurchase'] = first_purchase

    #Compute the number of seconds passed between the last purchase and the first purchase.
    rfm_score['LastPurchaseTimeDelta'] = rfm_score['LastPurchaseDate'] - rfm_score['FirstPurchase']
    rfm_score['LastPurchaseSeconds'] = rfm_score['LastPurchaseTimeDelta'].apply(lambda x : x.total_seconds())
    rfm_score = rfm_score.drop(columns=['LastPurchaseTimeDelta', 'FirstPurchase'])

    #Renormalize the 'number of seconds between last and first purchase' column to lie in (0,1)
    max_seconds = max(rfm_score['LastPurchaseSeconds'])
    min_seconds = min(rfm_score['LastPurchaseSeconds'])
    rfm_score['LastPurchaseSeconds'] -= min_seconds
    rfm_score['LastPurchaseSeconds'] /= (max_seconds - min_seconds)

    #Standardize number of purchases and money spent
    # rfm_score['NumberPurchases'] = standardize(rfm_score['NumberPurchases'])
    # rfm_score['MoneySpent'] = standardize(rfm_score['MoneySpent'])

    #Get the labels and the numerical values of RFM, rescaled to lie in (0,1).
    labels = kmeans.labels_
    rfm_display = rfm_score[['LastPurchaseDate', 'NumberPurchases', 'MoneySpent']]
    rfm_display = pd.concat([rfm_display, pd.Series(labels)], axis=1)
    rfm_display.columns = ['LastPurchaseDate', 'NumberPurchases', 'MoneySpent', 'Label']

    fig1 = px.scatter_3d(rfm_display, x='LastPurchaseDate', y='NumberPurchases', z='MoneySpent',
              color='Label')
    fig1.show()


    #Standardize number of purchases and money spent
    rfm_score['NumberPurchases'] = standardize(rfm_score['NumberPurchases'])
    rfm_score['MoneySpent'] = standardize(rfm_score['MoneySpent'])

    #Define rfm_numeric as the standardized version of LastPurchaseSeconds, NumberPurchases and MoneySpent
    rfm_numeric = rfm_score[['LastPurchaseSeconds', 'NumberPurchases', 'MoneySpent']]

    # Apply PCA to rfm_numeric to reduce to 2D
    projector = PCA(n_components=2)
    rfm_projected = projector.fit_transform(rfm_numeric)
    rfm_projected_list = []
    for i in range(rfm_projected.shape[0]):
        rfm_projected_list.append(rfm_projected[i])
    rfm_projected = pd.DataFrame(rfm_projected_list)
    rfm_projected = pd.concat([rfm_projected, pd.Series(labels)], axis = 1)
    rfm_projected.columns = ["x", "y", "Label"]

    fig2 = px.scatter(rfm_projected, x="x", y="y", color="Label")
    fig2.show()



    # #Plotly Express
    # fig = px.bar(
    #     data_frame=averages,
    #     x='State',
    #     y='Pct of Colonies Impacted',
    #     hover_data = ['State', 'Pct of Colonies Impacted'],
    # )



    return data1, columns1, data2, columns2, data3, columns3, fig1, fig2

#--------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)