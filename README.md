I got the dataset and the idea from https://machinelearningmastery.com/5-real-world-machine-learning-projects-you-can-build-this-weekend/?ref=dailydev.

In this project, I use KMeans clustering and DBSCAN to draw classify customers based on the order logs from an online retail website in 2011.

First, we segment customers according to their RFM scores (R = Recency, F = Frequency, M = Monetary Value). Each of the three scores is on a scale from 1 to 3.

For KMeans, we conclude by the elbow method that K=3 is the best parameter, and we visualize the clustered dataset in 3D and 2D (the latter after PCA). Usually, the 3 clusters are:
1. Recent shoppers (high R score, low F and M scores)
2. Heavy spenders (R score varies, F score usually high, M score almost always 3)
3. Former clients (low R score, F and M scores vary).

DBSCAN, on the other hand, performs very badly, since the data is too evenly spread out to create meaningful clusters by this method. I was not able to extract any useful information from it.
Anyone who spent a non-trivial amount of money is considered an outlier.

The Streamlit dashboard is available at https://onlineretailunsupervisedml.streamlit.app/.

To run the Dash Plotly dashboard, download the files "Online Retail.xlsx" and "retail_dashboard.py", open your terminal and type "python retail_dashboard.py". 
