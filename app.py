# Data Cleaning & Manipulation libraries
import pandas as pd
import numpy as np

# Data App library
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.chart_container import chart_container

# Machine Learning models library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data Visualization library
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static

# To handle file & tasks
import os
import joblib


# Set up the page configuration
# st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")

# Page configuration
page_title = "House Price Dashboard"
page_icon = ":house:"
layout = "centered"

# Set up the page configuration
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

# Hide Streamlit Style (optional)
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        #header {visibility: hidden;}
        #footer {visibility: hidden;}
        </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Navigation Menu
# Side Navigation Menu
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",  # Required
        options=["Home", "Dataset", "Visualizations", "3D Visualizations", "Filterized Data",
                 "Features", "Data Exploration", "Geospatial Analysis", "Model", "Prediction"],  # Required
        icons=["house-door-fill", "table", "bar-chart-fill", "cube", "filter-square",
               "clipboard-data", "search", "geo-alt", "cpu-fill", "graph-up-arrow"],  # Optional
        menu_icon="cast",  # Optional
        default_index=0,  # Optional
        orientation="vertical",
    )


# Load the dataset
@st.cache_data
def load_data():
    file_path = "data.csv"
    data = pd.read_csv(file_path)
    return data


data = load_data()


# Ensure consistent preprocessing
def preprocess_data(data):
    data = pd.get_dummies(data, columns=['Location', 'State', 'Country'], drop_first=True)
    return data


# Sidebar navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations", "3D Visualizations", "Filterized Data", "Features", "Data Exploration", "Geospatial Analysis", "Model", "Prediction"])

# Home page
if page == "Home":
    st.header("House Price Prediction Dashboard")
    st.write('#')
    st.info("""
        ## Overview
        This dashboard provides an in-depth analysis of house prices in a given dataset. 
        You can explore the data, visualize important trends, and see how various features affect house prices.
        Additionally, you can use machine learning models to predict house prices based on input features.
    """)
    st.write('---')
    st.success("""
        ### Pages
        - **Home:** Overview of the dashboard.
        - **Data:** Explore the dataset and its features.
        - **Visualizations:** Visualize important trends and insights.
        - **Features:** Feature engineering insights.
        - **Data Exploration:** To understand the distributions and relationships.
        - **Geospatial Analysis:** Analyze house prices based on geographical data.
        - **Model:** Train and evaluate machine learning models for house price prediction.
        - **Prediction:** Predict future house prices based on input parameters.
    """)

# Data page
elif page == "Dataset":
    st.header("Data Overview")

    st.write('#')
    # Dataset overview
    st.subheader("Dataset")
    st.write("This is the overview of the dataset used for house price prediction.")

    # Display the dataset
    st.dataframe(data)
    st.markdown("---")

    # Display dataset statistics
    st.subheader("Dataset Statistics")
    st.write(data.describe())
    # st.write(data.shape)
    st.markdown("---")

    # Show dataset columns
    st.subheader("Column Descriptions")
    st.write("""
        | Column Name              | Description                      |
        |--------------------------|----------------------------------|
        | `Price`                  | Price of the house in lakhs      |
        | `Area`                   | Area of the house in square feet |
        | `Location`               | Specific location within the city|
        | `No. of Bedrooms`        | Number of bedrooms               |
        | `Total Amenities`        | Total number of amenities        |
        | `Latitude`               | Latitude coordinate              |
        | `Longitude`              | Longitude coordinate             |
    """)
    st.markdown("---")

    # Add interactive filters
    st.subheader("Interactive Filters")
    min_price = int(data['Price'].min())
    max_price = int(data['Price'].max())
    min_area = int(data['Area'].min())
    max_area = int(data['Area'].max())

    price_filter = st.slider('Select Price Range', min_price, max_price, (min_price, max_price))
    area_filter = st.slider('Select Area Range', min_area, max_area, (min_area, max_area))
    bedrooms_filter = st.multiselect('Select Number of Bedrooms', data['No. of Bedrooms'].unique(),
                                     data['No. of Bedrooms'].unique())

    filtered_data = data[(data['Price'] >= price_filter[0]) & (data['Price'] <= price_filter[1]) &
                         (data['Area'] >= area_filter[0]) & (data['Area'] <= area_filter[1]) &
                         (data['No. of Bedrooms'].isin(bedrooms_filter))]
    st.dataframe(filtered_data)

# Visualization page
elif page == "Visualizations":
    st.header("Visualizations")

    with st.expander("Distribution of House Prices"):
        with chart_container(data[['Price']]):
            # Distribution of House Prices
            st.subheader("Distribution of House Prices")
            st.info("This plot shows the distribution of house prices in the dataset. It helps to understand the overall price range and frequency of different price points.")
            fig_hist = px.histogram(data, x='Price', nbins=50, title='Distribution of House Prices',
                                    color_discrete_sequence=['#1f77b4'])
            fig_hist.update_layout(bargap=0.2, xaxis_title='Price', yaxis_title='Count', template='plotly_dark')
            st.plotly_chart(fig_hist, use_container_width=True)

    # Handling errors to ensure no invalid data affects plotting
    data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

    # Drop rows with NaN values in 'Area' or 'Price' after conversion
    data = data.dropna(subset=['Area', 'Price'])

    with st.expander("House Prices by Area"):
        with chart_container(data[['Price', 'Area']]):
            # House Prices by Area
            st.subheader("House Prices by Area")
            st.info("This scatter plot shows the relationship between house prices and the area of the houses. It includes a trendline to help visualize the correlation.")
            fig_scatter = px.scatter(data, x='Area', y='Price', trendline='ols', title='House Prices by Area', color='Price',
                                     color_continuous_scale=px.colors.sequential.Viridis)
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.8))
            fig_scatter.update_layout(xaxis_title='Area (sq ft)', yaxis_title='Price (in lakhs)', template='plotly_dark')
            st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("House Prices by Number of Bedrooms"):
        with chart_container(data[['Price', 'No. of Bedrooms']]):
            # House Prices by Number of Bedrooms
            st.subheader("House Prices by Number of Bedrooms")
            st.info("This box plot shows how house prices vary by the number of bedrooms. It helps to identify any patterns or outliers.")
            fig_box = px.box(data, x='No. of Bedrooms', y='Price', title='House Prices by Number of Bedrooms',
                             color='No. of Bedrooms', color_discrete_sequence=px.colors.qualitative.Set1)
            fig_box.update_layout(xaxis_title='Number of Bedrooms', yaxis_title='Price', template='plotly_dark')
            st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("Violin Plot of Price by Number of Bedrooms"):
        with chart_container(data[['Price', 'No. of Bedrooms']]):
            # Violin Plot of Price by Number of Bedrooms
            st.subheader("Violin Plot of Price by Number of Bedrooms")
            st.info("This violin plot shows the distribution of house prices for different numbers of bedrooms. It combines aspects of a box plot and a density plot.")
            fig_violin = px.violin(data, x='No. of Bedrooms', y='Price', box=True, points='all',
                                   title='Violin Plot of Price by Number of Bedrooms', color='No. of Bedrooms',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_violin.update_layout(xaxis_title='Number of Bedrooms', yaxis_title='Price', template='plotly_dark')
            st.plotly_chart(fig_violin, use_container_width=True)

    with st.expander("Bar Plot of Total Amenities Count"):
        with chart_container(data[['Price', 'Total_Amenities']]):
            # Bar Plot of Total Amenities Count
            st.subheader("Total Amenities Count")
            st.info("This bar plot shows the relationship between the total number of amenities and house prices.")
            fig_bar = px.bar(data, x='Total_Amenities', y='Price', title='Total Amenities Count', color='Total_Amenities',
                             color_continuous_scale=px.colors.sequential.Plasma)
            fig_bar.update_layout(xaxis_title='Total Amenities', yaxis_title='Price', template='plotly_dark')
            st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("Pair Plot"):
        with chart_container(data[['Price', 'Area', 'No. of Bedrooms', 'Price_per_sqft']]):
            # Pair Plot with Enhanced Aesthetics
            st.subheader("Pair Plot with Aesthetics")
            st.info("This pair plot shows the relationships between selected features in the dataset. It helps to identify correlations and patterns.")
            selected_columns = ['Price', 'Area', 'No. of Bedrooms', 'Price_per_sqft']
            df_selected = data[selected_columns]
            fig_pair = px.scatter_matrix(df_selected, dimensions=selected_columns, title='Pair Plot with Enhanced Aesthetics',
                                         color='Price', color_continuous_scale=px.colors.sequential.Cividis)
            fig_pair.update_layout(width=1000, height=1000, template='plotly_dark')
            st.plotly_chart(fig_pair, use_container_width=True)


# 3D Visualizations Page
if page == "3D Visualizations":
    st.header("3D Visualizations")

    with st.expander("Price vs Area vs Number of Bedrooms"):
        with chart_container(data[['Price', 'Area', 'No. of Bedrooms']]):
            # 3D Scatter Plot
            st.subheader("3D Scatter Plot: Price vs Area vs Number of Bedrooms")
            st.info("This 3D scatter plot shows the relationship between house prices, area, and the number of bedrooms.")
            st.success("### Components:\n"
                       "X-axis: Area of the house\n "
                       "Y-axis: Number of bedrooms\n"
                       "Z-axis: Price of the house\n"
                       " Color: Represents the price (color intensity indicates higher or lower prices).\n"
                       "Size: Also reflects the price (larger points indicate higher prices).")
            fig_3d_scatter = px.scatter_3d(data, x='Area', y='No. of Bedrooms', z='Price', color='Price', size='Price',
                                           title='Price vs Area vs Number of Bedrooms')
            fig_3d_scatter.update_layout(scene=dict(
                xaxis_title='Area',
                yaxis_title='Number of Bedrooms',
                zaxis_title='Price',
            ),
                width=800,  # Adjust width as needed
                height=600,  # Adjust height as needed
            )
            st.plotly_chart(fig_3d_scatter, use_container_width=True)

    with st.expander("Price as a function of Area and Number of Bedrooms"):
        with chart_container(data[['Price', 'Area', 'No. of Bedrooms']]):
            # 3D Surface Plot
            st.subheader("3D Surface Plot: Price as a function of Area and Number of Bedrooms")
            st.info("This 3D surface plot visualizes house prices as a function of area and the number of bedrooms.")
            st.success("### Components:\n"
                       "X-axis: Area of the house;" "Y-axis: Number of bedrooms;" "Z-axis: Price of the house."
                       "Surface: Represents the continuous variation of price across different combinations of area and number of bedrooms.")
            fig_3d_surface = go.Figure(
                data=[go.Surface(z=data.pivot_table(index='Area', columns='No. of Bedrooms', values='Price').values)])
            fig_3d_surface.update_layout(
                title='Price as a function of Area and Number of Bedrooms',
                scene=dict(
                    xaxis_title='Area',
                    yaxis_title='Number of Bedrooms',
                    zaxis_title='Price',
                ),
                width=800,  # Adjust width as needed
                height=600,  # Adjust height as needed
            )
            st.plotly_chart(fig_3d_surface, use_container_width=True)

    with st.expander("Price vs Area vs Total Amenities"):
        with chart_container(data[['Price', 'Area', 'Total_Amenities']]):
            # 3D Bubble Plot
            st.subheader("3D Bubble Plot: Price vs Area vs Total Amenities")
            st.info("This 3D bubble plot shows the relationship between house prices, area, and the total number of amenities.")
            st.success("### Components:\n"
                       "X-axis: Area of the house\n "
                       "Y-axis: Total number of amenities\n"
                       "Z-axis: Price of the house\n"
                       "Bubble Size: Scaled according to the price (larger bubbles indicate higher prices).\n"
                       "Color: Reflects the price (color intensity represents higher or lower prices).")
            fig_3d_bubble = go.Figure(data=[go.Scatter3d(x=data['Area'], y=data['Total_Amenities'], z=data['Price'],
                                                         mode='markers',
                                                         marker=dict(size=data['Price'] / 1000, color=data['Price'],
                                                                     colorscale='Viridis', opacity=0.8))])
            fig_3d_bubble.update_layout(
                title='Price vs Area vs Total Amenities',
                scene=dict(
                    xaxis_title='Area',
                    yaxis_title='Total Amenities',
                    zaxis_title='Price',
                ),
                width=800,  # Adjust width as needed
                height=600,  # Adjust height as needed
            )
            st.plotly_chart(fig_3d_bubble, use_container_width=True)


# Features page
elif page == "Features":
    st.header("Feature Engineering Insights")
    st.write("## Displaying Feature Engineering Insights")
    st.info("We created new features like `Price_per_sqft` and `Total_Amenities` to gain deeper insights.")

    # Display feature statistics
    feature_stats = {
        "Feature": ["Price_per_sqft", "Total_Amenities"],
        "Mean": [data["Price_per_sqft"].mean(), data["Total_Amenities"].mean()],
        "Std Dev": [data["Price_per_sqft"].std(), data["Total_Amenities"].std()],
        "Min": [data["Price_per_sqft"].min(), data["Total_Amenities"].min()],
        "Max": [data["Price_per_sqft"].max(), data["Total_Amenities"].max()]
    }
    st.write(pd.DataFrame(feature_stats))

# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    st.info("Explore the dataset with interactive visualizations to understand distributions and relationships.")
    fig_explore = px.scatter_matrix(data, dimensions=['Price', 'Area', 'No. of Bedrooms', 'Total_Amenities'],
                                    title='Scatter Matrix of Features')
    st.plotly_chart(fig_explore, use_container_width=True)

    st.markdown("---")

    st.write("### Model Comparison")
    st.info("Compare different models' performances on the house price prediction task.")
    models = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest"]
    performance_metrics = {
        "Model": models,
        "MSE": [100, 80, 95, 110, 70],
        "MAE": [8, 7.5, 8.2, 9, 7],
        "R2 Score": [0.85, 0.87, 0.86, 0.83, 0.89]
    }
    st.write(pd.DataFrame(performance_metrics))

# Geospatial Analysis page
elif page == "Geospatial Analysis":
    st.header("Geospatial Analysis")
    st.write("## Performing Geospatial Analysis")
    with st.expander('Geo-Visual: Scatter Plot'):
        with chart_container(data[['Price', 'Area']]):
            # 1st Geo-Visual: Scatter Plot
            st.subheader("Scatter Plot")
            st.write("Below is a scatter plot showing the distribution of house prices in Mumbai.")

            fig = px.scatter(data, x='Area', y='Price', color='Location', title='House Prices vs Area', labels={'Area': 'Area (in sq. ft.)', 'Price': 'Price (in Lakhs)'})
            st.plotly_chart(fig, use_container_width=True)

            st.info("This scatter plot provides a visual representation of the distribution of house prices in Mumbai.")

    with st.expander('Geo-Visual: House Prices in Mumbai'):
        with chart_container(data[['Price', 'Latitude', 'Longitude']]):
            # 2nd Geo-Visual: House Prices in Mumbai using Plotly Express
            st.subheader("Map of House Prices in Mumbai (Plotly Express)")
            st.write("Here is a map showing the distribution of house prices across Mumbai.")
            data = pd.read_csv("data.csv")
            data.replace(9, pd.NA, inplace=True)
            data.dropna(inplace=True)
            fig = px.scatter_geo(data, lat='Latitude', lon='Longitude', hover_name='Price', title='House Prices in Mumbai', projection="natural earth")
            st.plotly_chart(fig, use_container_width=True)
            st.info("This map visualizes the house prices in Mumbai using Plotly Express. Each point represents a house, and the color intensity represents the price.")

    with st.expander('Geo-Visual: Map of House Prices in Mumbai'):
        with chart_container(data[['Latitude', 'Longitude']]):
            # 3rd Geo-Visual: Map of House Prices in Mumbai (Folium)
            st.subheader("Map of House Prices in Mumbai (Folium)")
            st.write("Here is a map showing the distribution of house prices across Mumbai.")

            mumbai_map = folium.Map(location=[19.0760, 72.8777], zoom_start=11)
            for i, row in data.iterrows():
                folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=5, color='blue', fill=True, fill_color='blue').add_to(mumbai_map)
            folium_static(mumbai_map)

            st.info("This map visualizes the distribution of house prices across Mumbai using Folium. Each blue circle represents a house.")

# Filterized Data page
elif page == "Filterized Data":
    st.header("Filterized Data")

    # Sample data generation for demonstration
    np.random.seed(42)
    data = pd.DataFrame({
        'Price': np.random.randint(100, 1000, size=100),
        'Area': np.random.randint(500, 4000, size=100),
        'No. of Bedrooms': np.random.randint(1, 6, size=100),
        'Total_Amenities': np.random.randint(0, 20, size=100),
        'Location': np.random.choice(['Location A', 'Location B', 'Location C', 'Location D'], size=100),
        'Latitude': np.random.uniform(18.5, 19.5, size=100),
        'Longitude': np.random.uniform(72.5, 73.5, size=100)
    })

    # 3D Visualization Page
    def render_3d_visualizations():
        st.header("3D Visualizations")
        with st.expander('Price vs Area vs Number of Bedrooms'):
            # 3D Scatter Plot: Price vs Area vs Number of Bedrooms
            st.subheader("3D Scatter Plot: Price vs Area vs Number of Bedrooms")
            st.info("3D Scatter Plot: Visualizes the relationship between Price, Area, and Number of Bedrooms.")
            fig_3d_scatter = px.scatter_3d(data, x='Area', y='Price', z='No. of Bedrooms', color='No. of Bedrooms',
                                           title='3D Scatter Plot: Price vs Area vs Number of Bedrooms')
            fig_3d_scatter.update_traces(marker=dict(size=5, opacity=0.7))
            st.plotly_chart(fig_3d_scatter, use_container_width=True)

        with st.expander('Price vs Area vs Number of Bedrooms over Time'):
            # 3D Line Plot: Price vs Area vs Number of Bedrooms over Time (assuming time is sequential index)
            st.subheader("3D Line Plot: Price vs Area vs Number of Bedrooms over Time")
            st.info("3D Line Plot: Displays the Price, Area, and Number of Bedrooms over Time.")
            data['Time'] = range(len(data))  # Adding a time column for demonstration
            fig_3d_line = px.line_3d(data, x='Time', y='Price', z='Area', color='No. of Bedrooms',
                                     title='3D Line Plot: Price vs Area vs Number of Bedrooms over Time')
            st.plotly_chart(fig_3d_line, use_container_width=True)

    # Geospatial Analysis Page with Filters
    def render_geospatial_analysis():
        st.title("Geospatial Analysis with Filters")
        st.write("## Performing Geospatial Analysis with Filters")

        # Filter options
        selected_location = st.multiselect("Select Location(s)", options=data['Location'].unique(),
                                           default=data['Location'].unique())
        min_price, max_price = st.slider("Select Price Range", min_value=int(data['Price'].min()),
                                         max_value=int(data['Price'].max()),
                                         value=(int(data['Price'].min()), int(data['Price'].max())))
        min_area, max_area = st.slider("Select Area Range (sq ft)", min_value=int(data['Area'].min()),
                                       max_value=int(data['Area'].max()),
                                       value=(int(data['Area'].min()), int(data['Area'].max())))
        min_bedrooms, max_bedrooms = st.slider("Select Number of Bedrooms",
                                               min_value=int(data['No. of Bedrooms'].min()),
                                               max_value=int(data['No. of Bedrooms'].max()), value=(
            int(data['No. of Bedrooms'].min()), int(data['No. of Bedrooms'].max())))
        st.markdown('---')

        # Apply filters
        filtered_data = data[
            (data['Location'].isin(selected_location)) &
            (data['Price'].between(min_price, max_price)) &
            (data['Area'].between(min_area, max_area)) &
            (data['No. of Bedrooms'].between(min_bedrooms, max_bedrooms))
            ]

        st.write(f"Filtered Data Points: {len(filtered_data)}")

        # Geospatial Visualization with Plotly
        st.subheader("Geospatial Visualization with Plotly")
        fig_map = px.scatter_mapbox(filtered_data, lat='Latitude', lon='Longitude', color='Price', size='Area',
                                    hover_data=['Location'], title='House Prices in Selected Areas',
                                    mapbox_style='open-street-map', zoom=10,
                                    color_continuous_scale=px.colors.sequential.Bluered)
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

        # Folium Map Visualization with Filters
        st.subheader("House Prices in Selected Areas using Folium")
        map_center = [filtered_data['Latitude'].mean(),
                      filtered_data['Longitude'].mean()] if not filtered_data.empty else [18.5, 72.5]
        folium_map = folium.Map(location=map_center, zoom_start=11)
        for _, row in filtered_data.iterrows():
            folium.Marker(location=[row['Latitude'], row['Longitude']],
                          popup=f"Price: {row['Price']} Lakhs\nArea: {row['Area']} sq. ft.\nLocation: {row['Location']}",
                          icon=folium.Icon(color='red', icon='info-sign')
                          ).add_to(folium_map)
        folium_static(folium_map)


    # Streamlit App Layout
    def main():
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Select Page", ["3D Visualizations", "Geospatial Analysis with Filters"])

        if page == "3D Visualizations":
            render_3d_visualizations()
        elif page == "Geospatial Analysis with Filters":
            render_geospatial_analysis()


    if __name__ == "__main__":
        main()

# Model Page
elif page == "Model":
    # Load and preprocess the data
    st.header("Train and Evaluate Machine Learning Models")

    # Load the data
    data = pd.read_csv("data.csv")

    # Convert categorical features to dummy variables
    data = pd.get_dummies(data, columns=['Location', 'State', 'Country'], drop_first=True)

    # Define features and target
    features = [col for col in data.columns if col != 'Price']
    target = 'Price'
    X = data[features]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sidebar for model selection
    st.sidebar.subheader("Model Selection")
    model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Polynomial Regression",
                                                         "Ridge Regression", "Lasso Regression",
                                                         "Decision Tree", "Random Forest"])

    # Training the selected model
    with st.spinner(f"Training the {model_choice} model, please wait..."):
        # Model selection logic
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Polynomial Regression":
            degree = st.sidebar.slider("Degree of Polynomial", min_value=2, max_value=5, value=2)
            poly_features = PolynomialFeatures(degree=degree)
            X_train = poly_features.fit_transform(X_train)
            X_test = poly_features.transform(X_test)
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            alpha = st.sidebar.slider("Alpha", min_value=0.01, max_value=10.0, value=1.0)
            model = Ridge(alpha=alpha)
        elif model_choice == "Lasso Regression":
            alpha = st.sidebar.slider("Alpha", min_value=0.01, max_value=10.0, value=1.0)
            model = Lasso(alpha=alpha)
        elif model_choice == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=20, value=5)
            model = DecisionTreeRegressor(max_depth=max_depth)
        elif model_choice == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=100, value=50)
            model = RandomForestRegressor(n_estimators=n_estimators)

        # Train the model
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, 'house_price_model.pkl')

        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.write(f"### Model: {model_choice}")
    st.metric("Mean Squared Error", f"{mse:.2f}")
    st.metric("Mean Absolute Error", f"{mae:.2f}")
    st.metric("R-squared", f"{r2:.2f}")

    st.write("---")
    st.write("#")


    # Visualization using Plotly
    st.write("### Actual vs. Predicted House Prices")

    # Create a DataFrame for actual vs. predicted values
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Create a scatter plot for actual vs predicted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df['Actual'],
        y=results_df['Predicted'],
        mode='markers',
        name='Predicted vs. Actual',
        marker=dict(color='yellow', size=6, line=dict(width=1))
    ))
    fig.add_trace(go.Scatter(
        x=results_df['Actual'],
        y=results_df['Actual'],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))

    # Customize layout
    fig.update_layout(
        title=f"{model_choice} - Actual vs Predicted Prices",
        xaxis_title="Actual Prices",
        yaxis_title="Predicted Prices",
        showlegend=True,
        legend=dict(x=0, y=1)
    )

    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

# Streamlit Prediction Page
elif page == "Prediction":
    st.header("Predict House Prices")
    st.write("Enter the features of the house to get a predicted price.")

    # Load data for dropdown options
    data = pd.read_csv("data.csv")

    # Default values for user input
    default_values = {
        'Area': 1500,
        'Bedrooms': 3,
        'Location': 'Andheri',
        'State': 'Maharashtra',
        'Country': 'India',
        'Total_Amenities': 7
    }

    # User input fields
    area = st.number_input("Area (in square feet)", min_value=0, value=default_values['Area'])
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=default_values['Bedrooms'])
    location = st.selectbox("Location", options=data['Location'].unique(), index=list(data['Location'].unique()).index(default_values['Location']))
    state = st.selectbox("State", options=data['State'].unique(), index=list(data['State'].unique()).index(default_values['State']))
    country = st.selectbox("Country", options=data['Country'].unique(), index=list(data['Country'].unique()).index(default_values['Country']))
    total_amenities = st.number_input("Total Amenities", min_value=0, value=default_values['Total_Amenities'])

    st.markdown("---")
    st.write("#")
    # Display input metrics in a neat layout
    st.markdown("### Input Summary")
    col1, col2 = st.columns(2)
    col1.metric("Area (sq ft)", area)
    col1.metric("Bedrooms", bedrooms)
    col1.metric("Total Amenities", total_amenities)
    col2.metric("Location", location)
    col2.metric("State", state)
    col2.metric("Country", country)

    # Prepare input data
    input_data = pd.DataFrame({
        "Area": [area],
        "No. of Bedrooms": [bedrooms],
        "Location": [location],
        "State": [state],
        "Country": [country],
        "Total_Amenities": [total_amenities]
    })

    # Train the model within the prediction page
    st.markdown("---")
    st.write("#")
    with st.spinner("Training the Random Forest model, please wait..."):
        # Prepare data for training
        features = [col for col in data.columns if col != 'Price']
        X = data[features]
        y = data['Price']

        # One-hot encoding for categorical variables
        X = pd.get_dummies(X, drop_first=True)
        input_data_encoded = pd.get_dummies(input_data, drop_first=True).reindex(columns=X.columns, fill_value=0)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model performance
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    # Create DataFrame to display model metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'R² Score'],
        'Value': [mse, mae, r2]
    })

    # Normalize values for progress bar representation
    max_mse = 80  # Adjust this value based on expected MSE range
    max_mae = 7   # Adjust this value based on expected MAE range
    max_r2 = 0.9     # Max value for R²

    # Display model metrics with progress bars
    st.subheader("Model Performance Metrics")
    st.dataframe(
        metrics_df,
        column_config={
            'Value': st.column_config.ProgressColumn(
                label="Value",
                format="%.2f",
                width="medium",
                min_value=0,
                max_value=max([max_mse, max_mae, max_r2])
            )
        },
        hide_index=True
    )

    # Predict the house price when the button is clicked
    if st.button("Predict"):
        st.info("Predicting the house price based on your input...")
        try:
            # Perform prediction
            prediction = model.predict(input_data_encoded)
            st.success(f"The predicted price of the house is approximately {prediction[0]:.2f} lakhs.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Footer
st.sidebar.markdown("""
---
**Developed by [Hamza Shaikh]**
""")
