import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


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



@st.cache_data
def load_data():
    base_dir = Path.cwd()
    return pd.read_excel(base_dir / "Online Retail.xlsx")


def standardize(column):
    min_value = min(column)
    max_value = max(column)
    column -= min_value
    column /= (max_value - min_value)
    return column


@st.cache_data  # Cache the clustering function
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans


retail = load_data()

st.title("Analysis of Online Retail Dataset - KMeans Clustering")

# Initialize session state variables if they don’t exist
if "num_clusters" not in st.session_state:
    st.session_state.num_clusters = 3  # Default cluster count

if "run_clustering" not in st.session_state:
    st.session_state.run_clustering = False  # Control when clustering runs

# Use text input instead of number_input to prevent reruns
n_clusters = int(st.text_input("Number of Clusters (Enter a number and click 'Run Clustering')", value=str(st.session_state.num_clusters)))

# Button to confirm selection
if st.button("Run Clustering"):
    try:
        st.session_state.num_clusters = int(n_clusters)  # Convert text input to integer
        st.session_state.run_clustering = True  # Allow clustering to run
    except ValueError:
        st.warning("Please enter a valid integer for the number of clusters.")

# Clustering code runs only when button is clicked
if st.session_state.run_clustering:
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
    kmeans = perform_clustering(rfm_df, n_clusters)

    # Generate all possible combinations of 1, 2, 3 in 3 columns
    combinations = pd.DataFrame([(x, y, z) for x in [1, 2, 3] for y in [1, 2, 3] for z in [1, 2, 3]],
                                columns=['col1', 'col2', 'col3'])
    combinations.columns = ['RecencyScore', 'FrequencyScore', 'MonetaryValueScore']

    # Predict the cluster for each combination
    combinations['Cluster'] = kmeans.predict(combinations)
    
    #Write the combinations dataframe with the cluster labels
    st.write("### RFM-Cluster Relationship")
    st.dataframe(combinations)

    # Print the combinations dataframe
    col1, col2 = st.columns(2)
    col1.write("### Cluster Names (Up to You!)")

    # Display an editable dataframe
    cluster_names = pd.DataFrame({
    "Description": ["" for i in range(n_clusters)]  # Editable column (initially empty)
    })
    cluster_names.index.name = "Cluster"
    edited_data = col1.data_editor(
        cluster_names,
        num_rows="fixed"  # Prevent adding/deleting rows
    )

    #We take the labels given by KMeans and turn them into a Series, that we then join as a column to rfm_score
    labelSeries = pd.Series(kmeans.labels_, name="Cluster")
    rfm_score.reset_index(drop=True, inplace=True)
    rfm_score_with_labels = pd.concat([rfm_score, labelSeries], axis=1)

    #Revenue by segment is then obtained by a simple groupby 
    revenue_by_segment = rfm_score_with_labels.groupby(["Cluster"]).MoneySpent.sum()
    col2.write("### Revenue per Cluster")
    col2.dataframe(revenue_by_segment)

    #Get the earliest purchase date
    first_purchase = min(retail['InvoiceDate'])
    rfm_score['FirstPurchase'] = first_purchase

    #Compute the number of seconds passed between the last purchase and the first purchase.
    rfm_score['LastPurchaseTimeDelta'] = rfm_score['InvoiceDate'] - rfm_score['FirstPurchase']
    rfm_score['LastPurchaseSeconds'] = rfm_score['LastPurchaseTimeDelta'].apply(lambda x : x.total_seconds())
    rfm_score = rfm_score.drop(columns=['LastPurchaseTimeDelta', 'FirstPurchase'])

    #Renormalize the 'number of seconds between last and first purchase' column to lie in (0,1)
    max_seconds = max(rfm_score['LastPurchaseSeconds'])
    min_seconds = min(rfm_score['LastPurchaseSeconds'])
    rfm_score['LastPurchaseSeconds'] -= min_seconds
    rfm_score['LastPurchaseSeconds'] /= (max_seconds - min_seconds)

    #Standardize number of purchases and money spent
    rfm_score['NumberPurchases'] = standardize(rfm_score['NumberPurchases'])
    rfm_score['MoneySpent'] = standardize(rfm_score['MoneySpent'])

    #Get the labels and the numerical values of RFM, rescaled to lie in (0,1).
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    rfm_numeric = rfm_score[['LastPurchaseSeconds', 'NumberPurchases', 'MoneySpent']]
    rfm_numeric = pd.concat([rfm_numeric,pd.Series(labels)], axis=1)
    rfm_numeric.columns = ['LastPurchaseSeconds', 'NumberPurchases', 'MoneySpent', 'Label']

    col3, col4 = st.columns(2)
    # Create a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Plot each cluster, one at a time
    for label in unique_labels:
        x_coords = rfm_numeric.loc[rfm_numeric['Label']==label, 'LastPurchaseSeconds'].to_list()
        y_coords = rfm_numeric.loc[rfm_numeric['Label']==label, 'NumberPurchases'].to_list()
        z_coords = rfm_numeric.loc[rfm_numeric['Label']==label, 'MoneySpent'].to_list()
        # Plot the points
        ax.scatter(
            x_coords, y_coords, z_coords,
            label=f'Cluster {label}',
        )

    # Add labels and legend
    ax.set_xlabel('Recency')
    ax.set_ylabel('Axis 2')
    ax.set_zlabel('Axis 3')
    plt.legend()
    col3.write("### 3D Plot of Clusters")
    col3.pyplot(fig)

    # Apply PCA to rfm_numeric to reduce to 2D
    projector = PCA(n_components=2)
    rfm_projected = projector.fit_transform(rfm_numeric)
    rfm_projected_list = []
    for i in range(rfm_projected.shape[0]):
        rfm_projected_list.append(rfm_projected[i])
    rfm_projected = pd.Series(rfm_projected_list)
    rfm_projected = pd.concat([rfm_projected, pd.Series(labels)], axis = 1)
    rfm_projected.columns = ["Coordinates", "Label"]

    # Create 2D figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Scatter the clusters in 2D
    for label in unique_labels:
        coords = rfm_projected.loc[rfm_projected['Label']==label, 'Coordinates']
        x_coords = []
        y_coords = []
        for point in coords:
            x_coords.append(point[0])
            y_coords.append(point[1])
        # Plot the points
        ax.scatter(
            x_coords, y_coords,
            label=f'Cluster {label}',
        )

    # Add labels and legend
    ax.set_xlabel('Axis 1')
    ax.set_ylabel('Axis 2')
    plt.legend()
    col4.write("### 2D Plot of Clusters after PCA")
    col4.pyplot(fig)

display_dataset = st.button("Display Online Retail Dataset")
if display_dataset:
    st.dataframe(retail)

