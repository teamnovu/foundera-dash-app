import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import pathlib


PATH = pathlib.Path(__file__).parent
df = pd.read_pickle(PATH.joinpath("grouped_by_canton.pkl"))

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    html.H1("Canton Data Viewer"),
    
    dcc.Dropdown(
        id='canton-selector',
        options=[{'label': i, 'value': i} for i in df.canton.drop_duplicates().sort_values()],
        value=['CH'],  # Default value
        multi=True,
    ),
    
    html.H3("Date range"),
    dcc.DatePickerRange(
        id='date-range-picker',
        start_date='2018-01-01',
        end_date='2024-12-31',
        min_date_allowed='2018-01-01',
        max_date_allowed='2024-12-31',
    ),

    html.H3("Rolling Average Window"),
    html.Div(    
        dcc.Slider(
            id='window-size-slider',
            min=1,
            max=14,
            value=14,  # Default value
            marks={i: str(i) for i in range(1, 15)},
            step=1,
            included=False,

        ), style={'max-width': '500px'},
    ),
    
    html.H3("Group by"),
    dcc.Dropdown(
        id='frequency-selector',
        options=[{'label': i.capitalize(), 'value': i} for i in ['day', 'week', 'month']],
        value='day',  # Default value
        style={'max-width': '500px'},
    ),
    
    html.H3("Normalize data?"),
    dcc.Dropdown(
        id='normalization-selector',
        options=[{'label': i.capitalize(), 'value': i} for i in ['normalized', 'raw data']],
        value='raw data',  # Default value
        style={'max-width': '500px'}
    ),
    dcc.RadioItems(id='trendline', options=['Show trendline', 'No trendline'], value='No trendline', inline=True),
    
    dcc.Graph(id='line-graph')
])

# Callback for updating the graph based on user inputs
@app.callback(
    Output('line-graph', 'figure'),
    [Input('canton-selector', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('window-size-slider', 'value'),
     Input('frequency-selector', 'value'),
    Input('normalization-selector', 'value'),
    Input('trendline', 'value')]
)
def update_graph(selected_cantons, start_date, end_date, window_size, by, normalized, trendline):
    # Filter data based on selections
    filtered_df = df[df['canton'].isin(selected_cantons) & 
                     (df['date'] >= start_date) & 
                     (df['date'] <= end_date) &
                    (df['by'] == by)]
    
    y_col = 'count' if window_size == 1 else f'count_rolling_{window_size}'

    def groupby(df):
        # normalize by first value in the time series
        start_value = df.dropna().sort_values(by='date').iloc[0][y_col]
        df[f'{y_col}_diff'] = (df[y_col] - start_value)
        df[f'{y_col}_normalized'] = df[f'{y_col}_diff'] / start_value
        return df[[y_col, f'{y_col}_diff', f'{y_col}_normalized', 'date']]
    normalized_df = filtered_df.groupby(["canton", 'by']).apply(groupby).reset_index()
    normalized_df = normalized_df.drop(columns=['level_2'])

    if normalized == 'normalized':
        data = normalized_df
        display_ycol = f'{y_col}_normalized'
        y_label = f"Normalized new foundings\n (in % relative to first {by})"
    else:
        data = filtered_df
        display_ycol = y_col
        y_label = f"New foundings per {by}"
        
    title = f'New foundings per {by}'
    if window_size > 1:
        title += f" ({window_size} {by} rolling average)"
        
    trendline = 'ols' if trendline == 'Show trendline' else None
    
    #fig = px.line(data, x="date", y=display_ycol, title=title, color='canton', labels={display_ycol: y_label})
    fig = px.scatter(data, x="date", y=display_ycol, title=title, color='canton',
             labels={display_ycol: y_label}, trendline=trendline)
    fig.update_traces(mode = 'lines')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
