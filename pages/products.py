from database.products import Table as ProductsTable
from database.shops import Table as ShopsTable
from .tools import sql_result_to_pandas
import plotly.express as px
import pandas as pd
import numpy as np


def sales_revenue_charts(st, product_table_df):
    sales_columns = product_table_df.columns[product_table_df.columns.str.contains(
        ' Sales')]
    revenue_columns = product_table_df.columns[product_table_df.columns.str.contains(
        ' Revenue')]
    dates = sorted(
        pd.to_datetime(
            sales_columns.str.split(' ').str[0],
            dayfirst=True)
    )

    df = pd.DataFrame(
        index=dates,
        data=np.c_[
            product_table_df[sales_columns].T.values,
            product_table_df[revenue_columns].T.values
        ],
        columns=[
            'Sales',
            'Revenue'
        ]
    )

    with st.sidebar:
        chart_type = st.radio(
            "Choose what to display",
            ('Sales', 'Revenue'))

    if chart_type == 'Sales':
        st.markdown("### Sales :chart_with_upwards_trend: chart")

        fig = px.line(
            df,
            x=df.index,
            y=df[df.columns[0]],
            markers=True,
            labels={
                'Sales': 'Sales',
                'index': 'Date'
            })
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### Revenue :chart: chart")

        fig = px.line(
            df,
            x=df.index,
            y=df[df.columns[1]],
            markers=True,
            labels={
                'Revenue': 'Revenue',
                'index': 'Date'
            })

        st.plotly_chart(fig, use_container_width=True)


def metrics(st, product_table_df):
    st.markdown("### Product metrics :currency_exchange:")

    max_price = product_table_df['Max price'].iloc[0]
    min_price = product_table_df['Min price'].iloc[0]
    average_price = product_table_df['Average price'].iloc[0]
    total_sales = product_table_df['Sales'].iloc[0]
    total_revenue = product_table_df['Revenue'].iloc[0]
    revenue_potential = product_table_df['Revenue potential'].iloc[0]
    lost_profit = product_table_df['Lost profit'].iloc[0]
    lost_profit_percent = product_table_df['Lost profit percent'].iloc[0]
    sales_per_day_average = product_table_df['Sales per day average'].iloc[0]

    column_1, column_2, column_3 = st.columns(3)

    column_1.metric("Max price", f"{max_price} $", "10 %")
    column_2.metric("Min price", f"{min_price} $", "8.55 %")
    column_3.metric("Average price", f"{average_price} $", "0.12 %")

    column_1.metric("Sales", f"{total_sales}", "-0.88 %")
    column_2.metric("Revenue", f"{total_revenue} $", "-0.86 %")
    column_3.metric("Revenue potential", f"{revenue_potential}", "40 %")

    column_1.metric("Lost profit", f"{lost_profit} $", "-10%")
    column_2.metric("Lost profit percent", f"{lost_profit_percent}", "8.55 %")
    column_3.metric("Sales per day average",
                    f"{sales_per_day_average}", "13.45 %")


def products_page(st, id):
    st.markdown(
        "# :shopping_trolley: Interactive map of places - Products report")

    table = ShopsTable().select_by_seller_id(id)
    table_df = sql_result_to_pandas(table, 'nodes')
    node_id = table_df['id'].iloc[0]

    products_table = ProductsTable().select_by_node_id(node_id)
    products_table_df = sql_result_to_pandas(products_table, 'products')

    st.markdown("### Products selection :clipboard:")
    product_name = st.selectbox(
        "Choose a product",
        tuple(products_table_df['Name'].values.tolist()))

    product_table_df = products_table_df[
        products_table_df['Name'] == product_name
    ].iloc[:1]

    with st.sidebar:
        option = st.selectbox(
            'Choose a chart?',
            ('Sales & Revenue charts', 'Metrics'))

        st.write('You selected:', option)

    if option == 'Sales & Revenue charts':
        sales_revenue_charts(st, product_table_df)
    elif option == 'Metrics':
        metrics(st, product_table_df)
