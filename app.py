from typing import List
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import dash
from dash import html, dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State

claim_df = pd.read_csv("./data/car_insurance_data_cleaned.csv")
claim_df["ClaimDate"] = pd.to_datetime(claim_df["ClaimDate"])
claim_df["AccidentDate"] = pd.to_datetime(claim_df["AccidentDate"])

# Format data for dashboard
claim_df.rename(columns={"Sex": "Gender", "FraudFound_P": "Fraud"}, inplace=True)

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Car Insurance Claim Analysis Dashboard"),
                html.P(
                    "Exploring Fraudulent and Non-Fraudulent Claims for Data-driven Insights and Risk Assessment"
                ),
            ],
            id="upper-container",
        ),
        html.Div(
            [html.H2("Overall Trends and Attribute Distributions")],
            id="distribution_header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(html.H2("Key Numbers")),
                                html.Div(
                                    [
                                        html.H3("Average Claim Size"),
                                        html.H4(f"${claim_df.ClaimSize.mean():,.2f}"),
                                    ],
                                    id="keynum_avg_claim",
                                ),
                                html.Div(
                                    [
                                        html.H3("Percentage of Fraud Cases"),
                                        html.H4(
                                            f'${(len(claim_df[claim_df["Fraud"] == 1]) / len(claim_df)) * 100:,.2f}%'
                                        ),
                                    ],
                                    id="keynum_perc_fraud",
                                ),
                                html.Div(
                                    [
                                        html.H3("Average Number of Claims per Year"),
                                        html.H4(
                                            f'{int(claim_df.groupby("Year")["PolicyNumber"].count().mean()):,.0f}'
                                        ),
                                    ],
                                    id="keynum_avg_count",
                                ),
                            ],
                            id="keynumbers",
                        ),
                    ],
                    id="infobox_upperleft",
                ),
                html.Div(
                    [
                        html.Div([dcc.Graph(id="timeline_graph")], id="timeline"),
                    ],
                    id="timelinebox",
                ),
            ],
            id="lower-container",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    html.H5("Distinguish between Fraud and Non-Fraud")
                                ),
                                html.Div(
                                    daq.BooleanSwitch(
                                        id="FraudBool",
                                        on=False,
                                    )
                                ),
                            ],
                            id="fraudbool_box",
                        ),
                        html.Div(
                            [
                                html.Div(html.H5("Pick a Date Range to display")),
                                html.Div(
                                    [
                                        dcc.DatePickerRange(
                                            id="datepicker_timeline",
                                            min_date_allowed=min(
                                                claim_df["AccidentDate"]
                                            ),
                                            max_date_allowed=max(claim_df["ClaimDate"]),
                                            end_date=max(claim_df["ClaimDate"]),
                                            start_date=min(claim_df["AccidentDate"]),
                                            clearable=False,
                                            day_size=30,
                                            number_of_months_shown=6,
                                            style={
                                                "backgroundColor": "red",
                                                "color": "white",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            id="datepicker_box",
                        ),
                        html.Div(
                            [
                                html.H5("Choose one Attribute"),
                                dcc.Dropdown(
                                    id="distributionattribute_select",
                                    options=[
                                        {"label": "Claim Size", "value": "ClaimSize"},
                                        {
                                            "label": "Age Of Policy Holder",
                                            "value": "AgeOfPolicyHolder",
                                        },
                                        {
                                            "label": "Driver Rating",
                                            "value": "DriverRating",
                                        },
                                        {
                                            "label": "Days Policy Accident",
                                            "value": "Days_Policy_Accident",
                                        },
                                        {
                                            "label": "Days Policy Claim",
                                            "value": "Days_Policy_Claim",
                                        },
                                        {
                                            "label": "Age of Vehicle",
                                            "value": "AgeOfVehicle",
                                        },
                                        {
                                            "label": "Number of Cars",
                                            "value": "NumberOfCars",
                                        },
                                        {
                                            "label": "Past Number of Claims",
                                            "value": "PastNumberOfClaims",
                                        },
                                        {"label": "Deductible", "value": "Deductible"},
                                        {
                                            "label": "NumberOfSuppliments",
                                            "value": "NumberOfSuppliments",
                                        },
                                        {"label": "Base Policy", "value": "BasePolicy"},
                                        {"label": "Fraud Found", "value": "Fraud"},
                                        {
                                            "label": "Marital Status",
                                            "value": "MaritalStatus",
                                        },
                                        {"label": "Fault", "value": "Fault"},
                                        {
                                            "label": "Vehicle Category",
                                            "value": "VehicleCategory",
                                        },
                                        {
                                            "label": "Police Report Filed",
                                            "value": "PoliceReportFiled",
                                        },
                                        {
                                            "label": "Witness Present",
                                            "value": "WitnessPresent",
                                        },
                                        {"label": "Agent Type", "value": "AgentType"},
                                        {
                                            "label": "Address Change Claim",
                                            "value": "AddressChange_Claim",
                                        },
                                        {"label": "Month", "value": "Month"},
                                        {
                                            "label": "Accident Area",
                                            "value": "AccidentArea",
                                        },
                                        {
                                            "label": "Month Claimed",
                                            "value": "MonthClaimed",
                                        },
                                        {"label": "Gender", "value": "Gender"},
                                        {
                                            "label": "Vehicle Price",
                                            "value": "VehiclePrice",
                                        },
                                        {"label": "Year", "value": "Year"},
                                        {"label": "Day of Week", "value": "DayOfWeek"},
                                        {
                                            "label": "Day of Week Claimed",
                                            "value": "DayOfWeekClaimed",
                                        },
                                    ],
                                    value="ClaimSize",
                                    clearable=False,
                                    searchable=False,
                                ),
                            ],
                            id="distributionattribute",
                        ),
                    ],
                    id="distribution_controls",
                ),
                html.Div([dcc.Graph(id="histogram_graph")], id="histogrambox"),
                html.Div([dcc.Graph(id="boxplot_graph")], id="boxplotbox"),
            ],
            id="distribution_graphs",
        ),
        html.Div(
            [html.H2("Relationship Between Attribute Pairs and Fraudulent Cases")],
            id="combination_header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Choose an Attribute"),
                        dcc.Dropdown(
                            id="combination_first_select",
                            options=[
                                {
                                    "label": "Age Of Policy Holder",
                                    "value": "AgeOfPolicyHolder",
                                },
                                {"label": "Driver Rating", "value": "DriverRating"},
                                {
                                    "label": "Days Policy Accident",
                                    "value": "Days_Policy_Accident",
                                },
                                {
                                    "label": "Days Policy Claim",
                                    "value": "Days_Policy_Claim",
                                },
                                {"label": "Age of Vehicle", "value": "AgeOfVehicle"},
                                {"label": "Number of Cars", "value": "NumberOfCars"},
                                {
                                    "label": "Past Number of Claims",
                                    "value": "PastNumberOfClaims",
                                },
                                {"label": "Deductible", "value": "Deductible"},
                                {
                                    "label": "NumberOfSuppliments",
                                    "value": "NumberOfSuppliments",
                                },
                                {"label": "Base Policy", "value": "BasePolicy"},
                                {"label": "Fraud Found", "value": "Fraud"},
                                {"label": "Marital Status", "value": "MaritalStatus"},
                                {"label": "Fault", "value": "Fault"},
                                {
                                    "label": "Vehicle Category",
                                    "value": "VehicleCategory",
                                },
                                {
                                    "label": "Police Report Filed",
                                    "value": "PoliceReportFiled",
                                },
                                {"label": "Witness Present", "value": "WitnessPresent"},
                                {"label": "Agent Type", "value": "AgentType"},
                                {
                                    "label": "Address Change Claim",
                                    "value": "AddressChange_Claim",
                                },
                                {"label": "Month", "value": "Month"},
                                {"label": "Accident Area", "value": "AccidentArea"},
                                {"label": "Month Claimed", "value": "MonthClaimed"},
                                {"label": "Gender", "value": "Gender"},
                                {"label": "Vehicle Price", "value": "VehiclePrice"},
                                {"label": "Year", "value": "Year"},
                                {"label": "Day of Week", "value": "DayOfWeek"},
                                {
                                    "label": "Day of Week Claimed",
                                    "value": "DayOfWeekClaimed",
                                },
                            ],
                            value="AgeOfVehicle",
                            clearable=True,
                            searchable=True,
                        ),
                    ],
                    id="combination_first_dropdown",
                ),
                html.Div(
                    [
                        html.H5("Choose a Second Attribute"),
                        dcc.Dropdown(
                            id="combination_second_select",
                            options=[
                                {
                                    "label": "Age Of Policy Holder",
                                    "value": "AgeOfPolicyHolder",
                                },
                                {"label": "Driver Rating", "value": "DriverRating"},
                                {
                                    "label": "Days Policy Accident",
                                    "value": "Days_Policy_Accident",
                                },
                                {
                                    "label": "Days Policy Claim",
                                    "value": "Days_Policy_Claim",
                                },
                                {"label": "Age of Vehicle", "value": "AgeOfVehicle"},
                                {"label": "Number of Cars", "value": "NumberOfCars"},
                                {
                                    "label": "Past Number of Claims",
                                    "value": "PastNumberOfClaims",
                                },
                                {"label": "Deductible", "value": "Deductible"},
                                {
                                    "label": "NumberOfSuppliments",
                                    "value": "NumberOfSuppliments",
                                },
                                {"label": "Base Policy", "value": "BasePolicy"},
                                {"label": "Fraud Found", "value": "Fraud"},
                                {"label": "Marital Status", "value": "MaritalStatus"},
                                {"label": "Fault", "value": "Fault"},
                                {
                                    "label": "Vehicle Category",
                                    "value": "VehicleCategory",
                                },
                                {
                                    "label": "Police Report Filed",
                                    "value": "PoliceReportFiled",
                                },
                                {"label": "Witness Present", "value": "WitnessPresent"},
                                {"label": "Agent Type", "value": "AgentType"},
                                {
                                    "label": "Address Change Claim",
                                    "value": "AddressChange_Claim",
                                },
                                {"label": "Month", "value": "Month"},
                                {"label": "Accident Area", "value": "AccidentArea"},
                                {"label": "Month Claimed", "value": "MonthClaimed"},
                                {"label": "Gender", "value": "Gender"},
                                {"label": "Vehicle Price", "value": "VehiclePrice"},
                                {"label": "Year", "value": "Year"},
                                {"label": "Day of Week", "value": "DayOfWeek"},
                                {
                                    "label": "Day of Week Claimed",
                                    "value": "DayOfWeekClaimed",
                                },
                            ],
                            value="AgeOfPolicyHolder",
                            clearable=True,
                            searchable=True,
                        ),
                    ],
                    id="combination_second_dropdown",
                ),
            ],
            id="combination_control_box",
        ),
        html.Div([dcc.Graph(id="stack_graph")], id="stack"),
        html.Div([dcc.Graph(id="heatmap_graph")], id="heatmap"),
        html.Div(
            [html.H2("Multivariate Attribute Patterns for Fraud Identification")],
            id="multivariate_header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Select Attributes"),
                        dcc.Dropdown(
                            id="parallel_multi_select",
                            options=[
                                {"label": col, "value": col}
                                for col in claim_df.select_dtypes(
                                    include=[np.number]
                                ).columns
                                if col not in ["Fraud", "Unnamed: 0"]
                            ],
                            value=["ClaimSize", "DriverRating"],
                            multi=True,
                            clearable=True,
                            searchable=True,
                        ),
                    ],
                    id="parallel_dropdown",
                )
            ],
            id="parallel_dropdownbox",
        ),
        html.Div(
            [
                html.Div([dcc.Graph(id="parallel_graph")], id="parallel"),
            ],
            id="parallelbox",
        ),
    ]
)


@app.callback(
    Output("timeline_graph", "figure"),
    Input("datepicker_timeline", "start_date"),
    Input("datepicker_timeline", "end_date"),
    Input("FraudBool", "on"),
)
def render_content(start_date, end_date, fraud_toggle):
    """Renders a plotly histogram of the number of claims over time, with the option to split by fraudulent/non-fraudulent claims.

    Args:
        start_date (str): Start date of the timeline in YYYY-MM-DD format.
        end_date (str): End date of the timeline in YYYY-MM-DD format.
        fraud_toggle (bool): Toggle to include fraudulent claims in the plot.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object.
    """
    mask = (claim_df.ClaimDate > pd.to_datetime(start_date)) & (
            claim_df.ClaimDate < pd.to_datetime(end_date)
    )

    # Initiate Figure
    fig = go.Figure()
    if fraud_toggle:
        # Add histogram trace for fraudulent claims
        fig.add_trace(
            go.Histogram(
                x=claim_df[mask & (claim_df["Fraud"] == 1)]["AccidentDate"],
                name="Fraudulent Claims",
                marker=dict(color="#FF6666"),
                histfunc="count",
            )
        )
        # Add histogram trace for non-fraudulent claims
        fig.add_trace(
            go.Histogram(
                x=claim_df[mask & (claim_df["Fraud"] == 0)]["ClaimDate"],
                name="Non-Fraudulent Claims",
                marker=dict(color="#1C364D"),
                histfunc="count",
            )
        )
        title = "Number of Fraudulent and Non-Fraudulent Claims Over Time"
    else:
        fig.add_trace(
            go.Histogram(
                x=claim_df[mask]["ClaimDate"],
                name="Fraudulent Claims",
                marker=dict(color="#1C364D"),
                histfunc="count",
            )
        )
        title = "Number of Claims over Time"
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Claims",
        title_text=title,
        title_x=0.5,
        barmode="stack",
    )

    return fig


@app.callback(
    Output("histogram_graph", "figure"),
    Input("distributionattribute_select", "value"),
    Input("FraudBool", "on"),
)
def plot_histogram(attribute: str, fraud_bool: bool) -> go.Figure:
    """Plot a histogram of the given attribute in the claim data frame, optionally split by fraud status.
    Args:
        attribute (str): The name of the attribute to plot.
        fraud_bool (bool): If True, split the histogram by fraud status.
    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure object containing the histogram.
    """
    if fraud_bool:
        # Create the histogram trace for the overall count
        no_fraud_trace = go.Histogram(
            x=claim_df[claim_df["Fraud"] == 0][attribute],
            nbinsx=30,
            name="No Fraud Cases",
            marker=dict(color="#1C364D"),
            histfunc="count",
        )

        # Create the histogram trace for the count of fraud cases
        fraud_trace = go.Histogram(
            x=claim_df[claim_df["Fraud"] == 1][attribute],
            nbinsx=30,
            name="Fraud Cases",
            marker=dict(color="#FF6666"),
            histfunc="count",
        )
        fig = go.Figure(data=[fraud_trace, no_fraud_trace])
        title = f"Histogram of {attribute} Distribution in Fraudulent and Non-Fraudulent Claims"
    else:
        overall_trace = go.Histogram(
            x=claim_df[attribute],
            nbinsx=30,
            marker=dict(color="#1C364D"),
            histfunc="count",
        )
        fig = go.Figure(data=[overall_trace])
        title = f"Histogram of {attribute} Distribution"
    # Create the figure and update the layout

    fig.update_layout(
        xaxis_title=attribute,
        yaxis_title="Count",
        title_text=title,
        barmode="stack",
    )

    return fig


@app.callback(
    Output("boxplot_graph", "figure"),
    Input("distributionattribute_select", "value"),
    Input("FraudBool", "on"),
)
def update_boxplot(attribute, fraud_bool):
    """Update a box plot of the given attribute in the claim data frame, optionally split by fraud status.

    Args:
        attribute (str): The name of the attribute to plot.
        fraud_bool (bool): If True, split the box plot by fraud status.
    Returns:
        go.Figure: A Plotly Figure object representing the updated box plot.
    """
    if fraud_bool:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

        fraud_trace = px.box(
            claim_df[claim_df["Fraud"] == 1],
            y=attribute,
            points="all",
        ).data[0]

        nonfraud_trace = px.box(
            claim_df[claim_df["Fraud"] == 0],
            y=attribute,
            points="all",
        ).data[0]

        fraud_trace.line.color = "#FF6666"
        fraud_trace.marker.color = "#FF6666"

        nonfraud_trace.line.color = "#1C364D"
        nonfraud_trace.marker.color = "#1C364D"

        fig.add_trace(fraud_trace, row=1, col=1)
        fig.add_trace(nonfraud_trace, row=1, col=2)

        # Set titles for subplots
        fig.update_xaxes(title_text="Fraudulent", row=1, col=1)
        fig.update_xaxes(title_text="Non-Fraudulent", row=1, col=2)
        title = f"Boxplot of {attribute} Distribution in Fraudulent and Non-Fraudulent Claims"
    else:
        fig = px.box(
            claim_df,
            y=attribute,
            points="all",
        )
        fig.update_traces(line_color="#1C364D", marker_color="#1C364D")

        title = f"Boxplot of {attribute} Distribution"

    # Set layout
    fig.update_layout(title=title, yaxis=dict(title=attribute))

    return fig


@app.callback(
    Output("heatmap_graph", "figure"),
    Input("combination_first_select", "value"),
    Input("combination_second_select", "value"),
)
def update_heatmap(attribute_x: str, attribute_y: str) -> go.Figure:
    """Update a heatmap of fraud percentage by the combination of two attributes.

    Args:
        attribute_x (str): The name of the first attribute to plot.
        attribute_y (str): The name of the second attribute to plot.

    Returns:
        go.Figure: A Plotly Figure object representing the updated heatmap.
    """
    # Group the data and calculate fraud percentage
    grouped_data = claim_df.groupby([attribute_y, attribute_x])["Fraud"].mean() * 100

    # Create a pivot table
    pivot_table = grouped_data.unstack()

    # Define the primary colors for the colormap
    blue_color = "#1C364D"
    red_color = "#E57373"

    # Create the custom colormap
    colormap = [
        [0.0, "rgb(220, 220, 220)"],  # Light gray for neutral values
        [0.0, blue_color],  # Start with blue color for low values
        [1.0, red_color],
    ]  # End with red color for high values

    # Create the heatmap using the pivot table
    heatmap = go.Heatmap(
        x=pivot_table.columns,
        y=pivot_table.index,
        z=pivot_table.values,
        colorscale=colormap,  # Choose a suitable color scale
        colorbar=dict(title="Fraud Percentage"),
    )

    # Set the layout for the heatmap
    layout = go.Layout(
        title=f"Fraud Distribution by {attribute_x} and {attribute_y}",
        xaxis=dict(title=attribute_x),
        yaxis=dict(title=attribute_y),
    )

    # Create the figure and add the heatmap
    fig = go.Figure(data=[heatmap], layout=layout)
    fig.data[0].update(zmin=0, zmax=100)
    return fig


@app.callback(
    Output("stack_graph", "figure"),
    Input("combination_first_select", "value"),
    Input("combination_second_select", "value"),
)
def update_stack(attribute_x: str, attribute_stack: str) -> go.Figure:
    """Update a stacked bar plot of the number of fraud cases by the combination of two attributes.

    Args:
        attribute_x (str): The name of the first attribute to plot.
        attribute_stack (str): The name of the second attribute to stack.

    Returns:
        go.Figure: A Plotly Figure object representing the updated stacked bar plot.
    """
    # Group data by the selected attributes and count the number of fraud cases
    data = claim_df.groupby([attribute_x, attribute_stack])["Fraud"].sum().unstack()

    # Get unique stack values from both the DataFrame and the 'AgeOfPolicyHolder' column
    stacks = set(data.columns.tolist() + claim_df[attribute_stack].unique().tolist())

    # Generate a categorical colormap based on the unique stack values
    colormap = dict(zip(stacks, px.colors.sequential.Burgyl))

    # Create stacked bar plot with colormap
    fig = go.Figure(
        data=[
            go.Bar(
                x=data.index,
                y=data[column],
                name=column,
                marker=dict(
                    color=colormap.get(column, "lightgray")
                ),  # Assign color from the colormap
            )
            for column in data.columns
        ]
    )

    # Set layout
    fig.update_layout(
        title=f"Number of Fraud Cases by {attribute_x} and {attribute_stack}",
        xaxis=dict(title=attribute_x),
        yaxis=dict(title="Number of Fraud Cases"),
        barmode="stack",
    )

    return fig


@app.callback(
    Output("parallel_graph", "figure"), Input("parallel_multi_select", "value")
)
def update_parallel_plot(attributes: List[str]) -> go.Figure:
    """Update a parallel coordinates plot of the selected attributes and fraud status.

    Args:
        attributes (List[str]): A list of attribute names to plot.

    Returns:
        go.Figure: A Plotly Figure object representing the updated parallel coordinates plot.
    """
    # Define the primary colors for the colormap
    blue_color = "#1C364D"
    red_color = "#E57373"

    # Create the custom colormap
    colormap = [
        [0.0, "rgb(220, 220, 220)"],  # Light gray for neutral values
        [0.0, blue_color],  # Start with blue color for low values
        [1.0, red_color],
    ]  # End with red color for high values

    fig = px.parallel_coordinates(
        claim_df,
        color_continuous_scale=colormap,
        color="Fraud",
        dimensions=attributes + ["Fraud"],
    )
    fig.update_layout(
        title="Parallel Plot for Fraud Status and Attribute Comparison", height=600
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
