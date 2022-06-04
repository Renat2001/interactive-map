# from pages.malls import malls_page
# from pages.shops import shops_page
# from pages.products import products_page
# from pages.search_logs import search_logs

import streamlit as st

st.set_page_config(layout="wide")

query_params = st.experimental_get_query_params()
id = query_params['id'][0]
name = query_params['name'][0]

# with open('user_id.txt', 'r') as file:
#     data = file.read().replace('\n', '=').split('=')
# id = int(data[1])
# name = data[-1]


with st.sidebar:
    st.markdown(f"### :alien: {name}")

    if id == 1:
        option = st.selectbox(
            'Choose preferred page',
            ('Malls', 'Search logs'))
    else:
        option = st.selectbox(
            'Choose preferred page',
            ('Shops', 'Products', 'Search logs'))

    st.write('You selected:', option)

if option == 'Malls':
    # malls_page(st)
    pass
elif option == 'Shops':
    # shops_page(st, id)
    pass
elif option == 'Products':
    # products_page(st, id)
    pass
else:
    # search_logs(st)
    pass
