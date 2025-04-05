import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from datetime import datetime
from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series import FeatureTransformer, DefaultFeatures, TabPFNTimeSeriesPredictor, TabPFNMode
import webbrowser
import threading
import time

# --- Configuration ---
COLORS = {
    'background': '#f9f9f9',
    'card_background': '#ffffff',
    'primary': '#3366CC',
    'secondary': '#6699CC',
    'accent': '#FF6666',
    'text': '#333333',
    'light_text': '#666666',
    'border': '#e0e0e0',
    'grid': '#f0f0f0',
    'fillcolor': 'rgba(51, 102, 204, 0.15)'
}

PREDICTION_LENGTH = 30
DATA_FILE = 'synthetic_electronics_single_country_sales_LONG_2024_2025.csv'

PROMOTION_MAPPING = {
    'No Promotion': 0,
    'SpringSale': 1,
    'SummerDeals': 1,
    'Back2School': 1,
    'BlackFridayWeek': 1,
    'HolidaySale': 1,
    'YearEndClearance': 1
}

# --- Data Processing Functions ---
def load_and_preprocess_data(file_path):
    """Load and preprocess the sales data."""
    df = pd.read_csv(file_path)
    
    # Clean and transform data
    df['Category'] = df['Category'].replace({'AirSolutions': 'Air Solutions', 'HomeAppliances': 'Home Appliances'})
    df['ActivePromotion'] = df['ActivePromotion'].fillna('No Promotion')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Aggregate sales by category, timestamp, and promotion
    df = df.groupby(['Category', 'Timestamp', 'ActivePromotion'])['Sales(USD)'].sum()
    df = df.reset_index()
    
    # Map promotions to binary values
    df['ActivePromotion'] = df['ActivePromotion'].map(PROMOTION_MAPPING)
    
    # Rename columns for model compatibility
    df = df.rename(columns={'Sales(USD)': 'target', 'ActivePromotion': 'Promotion'})
    
    return df

def prepare_time_series_data(df):
    """Convert dataframe to TimeSeriesDataFrame format."""
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='Category',
        timestamp_column='Timestamp',
    )
    return tsdf

def generate_features_and_predictions(tsdf, prediction_length):
    """Generate features and predictions using TabPFN."""
    # Generate test data for future predictions
    future_test_tsdf = generate_test_X(tsdf, prediction_length=prediction_length)
    
    # Select and add features
    selected_features = [
        DefaultFeatures.add_running_index,
        DefaultFeatures.add_calendar_features
    ]
    
    full_train_tsdf, future_test_tsdf = FeatureTransformer.add_features(
        tsdf, future_test_tsdf, selected_features
    )
    
    # Initialize predictor and make predictions
    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
    future_predictions = predictor.predict(full_train_tsdf, future_test_tsdf)
    
    # Reset index to turn multi-index into columns
    forecast_df = future_predictions.reset_index()
    
    return forecast_df

# --- Dashboard Components ---
def create_layout():
    """Create the Dash app layout."""
    return html.Div([
        html.Div([
            # Header
            html.H1("30-Day Sales Forecast", style={
                'fontFamily': 'Helvetica, Arial, sans-serif',
                'fontWeight': '300',
                'color': COLORS['text'],
                'marginBottom': '20px',
                'paddingBottom': '10px',
                'borderBottom': f'1px solid {COLORS["border"]}'
            }),
            
            # Category Selector
            html.Div([
                html.Div([
                    html.Label("Select Product Category:", style={
                        'fontWeight': '500',
                        'marginBottom': '8px',
                        'display': 'block',
                        'color': COLORS['light_text']
                    }),
                    dcc.Dropdown(
                        id='item-dropdown',
                        options=[{'label': item, 'value': item} for item in 
                                ['Air Solutions', 'Audio', 'Home Appliances', 'IT', 'Mobile', 'TVs']],
                        value='Air Solutions',
                        clearable=False,
                        style={
                            'borderRadius': '4px',
                            'border': f'1px solid {COLORS["border"]}',
                        }
                    )
                ], style={'width': '300px', 'marginBottom': '20px'})
            ]),
            
            # Forecast Graph
            html.Div([
                dcc.Graph(
                    id='forecast-graph',
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [
                            'select2d', 'lasso2d', 'autoScale2d',
                            'toggleSpikelines'
                        ]
                    }
                )
            ], style={
                'backgroundColor': COLORS['card_background'],
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)',
                'padding': '15px',
                'marginBottom': '20px'
            }),
            
            # Forecast Statistics
            html.Div([
                html.H3("Forecast Statistics", style={
                    'fontFamily': 'Helvetica, Arial, sans-serif',
                    'fontWeight': '400',
                    'color': COLORS['text'],
                    'marginBottom': '15px',
                    'paddingBottom': '8px',
                    'borderBottom': f'1px solid {COLORS["border"]}'
                }),
                html.Div(id='forecast-stats')
            ], style={
                'backgroundColor': COLORS['card_background'],
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)',
                'padding': '20px',
            })
        ], style={
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '20px',
            'backgroundColor': COLORS['background'],
            'fontFamily': 'Helvetica, Arial, sans-serif',
        })
    ])

def create_forecast_figure(filtered_df, selected_item):
    """Create a Plotly figure for the forecast."""
    fig = go.Figure()

    # Add the lower quantile (0.1) as the lower boundary of uncertainty
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df[0.1],
        mode='lines',
        name='Lower Q10',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add the upper quantile (0.9) as the upper boundary of uncertainty
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df[0.9],
        mode='lines',
        name='Upper Q90',
        line=dict(width=0),
        fillcolor=COLORS['fillcolor'],
        fill='tonexty',
        showlegend=False
    ))
    
    # Add the median forecast line (0.5 quantile)
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df[0.5],
        mode='lines+markers',
        name='Forecast',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, color=COLORS['primary'], line=dict(width=2, color='white'))
    ))
    
    # Custom hover template to format numbers with fewer decimals
    hovertemplate = (
        '<b>%{x|%b %d, %Y}</b><br>' +
        'Forecast: $%{y:,.0f}<br>' +
        '<extra></extra>'
    )
    
    # Update traces with custom hover template
    for trace in fig.data:
        if trace.name == "Forecast":
            trace.hovertemplate = hovertemplate
        else:
            trace.hovertemplate = None
    
    # Update the layout of the figure for a modern look
    fig.update_layout(
        title={
            'text': f'30-Day Sales Forecast for {selected_item}',
            'font': {
                'family': 'Helvetica, Arial, sans-serif',
                'size': 22,
                'color': COLORS['text']
            },
            'x': 0.01,
            'y': 0.95
        },
        xaxis={
            'title': None,
            'gridcolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
            'tickformat': '%b %d',
        },
        yaxis={
            'title': 'Forecasted Sales ($)',
            'gridcolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
            'title_font': {'color': COLORS['light_text']},
            'tickprefix': '$',
            'tickformat': ',.0f',
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        hoverlabel={
            'bgcolor': 'white',
            'font_size': 14,
            'font_family': 'Helvetica, Arial, sans-serif'
        },
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': COLORS['border'],
            'borderwidth': 1
        },
        margin={'l': 40, 'r': 40, 't': 80, 'b': 40}
    )
    
    # Add subtle grid lines to help with readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    
    return fig

def calculate_forecast_stats(filtered_df):
    """Calculate statistics for the forecast."""
    avg_forecast = filtered_df[0.5].mean()
    min_forecast = filtered_df[0.5].min()
    max_forecast = filtered_df[0.5].max()
    total_forecast = filtered_df[0.5].sum()
    avg_uncertainty = (filtered_df[0.9] - filtered_df[0.1]).mean()
    
    # Create a modern, card-style stats table
    stats = html.Div([
        html.Div([
            html.Div([
                html.Div("Average Forecast", className="stat-label"),
                html.Div(f"${avg_forecast:,.0f}", className="stat-value")
            ], className="stat-card"),
            
            html.Div([
                html.Div("Minimum Forecast", className="stat-label"),
                html.Div(f"${min_forecast:,.0f}", className="stat-value")
            ], className="stat-card"),
            
            html.Div([
                html.Div("Maximum Forecast", className="stat-label"),
                html.Div(f"${max_forecast:,.0f}", className="stat-value")
            ], className="stat-card"),
            
            html.Div([
                html.Div("Total 30-Day Forecast", className="stat-label"),
                html.Div(f"${total_forecast:,.0f}", className="stat-value")
            ], className="stat-card"),
            
            html.Div([
                html.Div("Avg. Uncertainty Range", className="stat-label"),
                html.Div(f"${avg_uncertainty:,.0f}", className="stat-value")
            ], className="stat-card"),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fill, minmax(200px, 1fr))',
            'gap': '20px',
        })
    ])
    
    return stats

def create_custom_css():
    """Create custom CSS for the dashboard."""
    return '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Sales Forecast Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .stat-card {
                background-color: white;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                text-align: center;
                border-top: 3px solid #3366CC;
            }
            .stat-label {
                color: #666666;
                font-size: 14px;
                margin-bottom: 8px;
            }
            .stat-value {
                color: #333333;
                font-size: 24px;
                font-weight: 500;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- Main Application ---
def main():
    """Main function to run the application."""
    # Process data
    print("Loading and processing data...")
    df = load_and_preprocess_data(DATA_FILE)
    tsdf = prepare_time_series_data(df)
    
    print("Generating features and predictions...")
    forecast_df = generate_features_and_predictions(tsdf, PREDICTION_LENGTH)
    print("Model training and prediction completed.")
    
    # Initialize Dash app
    app = Dash(__name__)
    app.layout = create_layout()
    app.index_string = create_custom_css()
    
    # Define callback for interactivity
    @app.callback(
        [Output('forecast-graph', 'figure'),
         Output('forecast-stats', 'children')],
        Input('item-dropdown', 'value')
    )
    def update_graph(selected_item):
        # Filter the DataFrame to the selected item
        filtered_df = forecast_df[forecast_df['item_id'] == selected_item]
        
        # Create figure and calculate stats
        fig = create_forecast_figure(filtered_df, selected_item)
        stats = calculate_forecast_stats(filtered_df)
        
        return fig, stats
    
    # Open browser after a delay to ensure server is ready
    def open_browser():
        """Open browser after giving the server time to start."""
        print("Waiting for server to initialize...")
        time.sleep(15)  # Wait 2 seconds for the server to start
        print("Opening browser at http://127.0.0.1:8050/")
        webbrowser.open("http://127.0.0.1:8050/")
    
    # Start the browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Run the app (this blocks until the app is closed)
    print("Starting server...")
    app.run(debug=False, port=8050)

if __name__ == '__main__':
    main()