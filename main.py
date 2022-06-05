from malls import malls_page
from shops import shops_page
from products import products_page
from search_logs import search_logs_page
from nlp_search import nlp_search_page

import streamlit as st

st.set_page_config(layout="wide")

query_params = st.experimental_get_query_params()

try:
    id = int(query_params['id'][0])
    name = query_params['name'][0]
except:
    id = 1
    name = "Admin"

with st.sidebar:
    st.markdown(f"### :alien: {name}")

    if id == 1:
        option = st.selectbox(
            'Choose preferred page',
            ('Malls', 'Search logs', 'NLP Search'))
    else:
        option = st.selectbox(
            'Choose preferred page',
            ('Shops', 'Products', 'Search logs', 'NLP Search'))

    st.write('You selected:', option)

if option == 'Malls':
    malls_page(st)
elif option == 'Shops':
    shops_page(st, id)
elif option == 'Products':
    products_page(st, id)
elif option == 'NLP Search':
    nlp_search_page(st)
else:
    search_logs_page(st)
