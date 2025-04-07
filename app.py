import streamlit as st
import pandas as pd
import plotly.express as px


def get_uplift(data, a, b):
    c = 1 - a - b
    model_coeffs = pd.DataFrame.from_dict({'Кредитные карты': [a], 'Микрокредиты': [b], 'Кредиты': [c]},orient='index').reset_index()
    model_coeffs.columns =['product_type', 'coef']

    data = data.merge(model_coeffs, on ='product_type')
    data['score'] = data['score'] * data['coef']

    data['rank_pred'] = data.groupby('request_calc_id')['score'].transform(lambda x: x.rank(ascending=False, method='first'))

    data = data.merge(data.loc[:, ['request_calc_id', 'product_type', 'bank_id', 'revenue', 'rank_pred', 'score']], 
                    how='inner', 
                    left_on=['request_calc_id', 'rank_true'],
                    right_on = ['request_calc_id', 'rank_pred'],
                    suffixes=(None, '_new')
                    )
    
    data = data[data.score > 0]
    new_revenue = data[data['target'] == 1].loc[:, 'revenue_new'].sum()

    return (new_revenue, data)


# Или можно сделать простую заглушку
data = pd.read_csv('data/20250407_data4streamlit.csv')


# Интерфейс
st.title("Интерактивный расчёт get_uplift")
st.markdown("Выберите параметры `a` и `b`, и получите значение функции `get_uplift()`.")

a = st.slider("Модель КК", 0.0, 1.0, 0.5, 0.01)
b = st.slider("Модель МФО", 0.0, 1.0, 0.5, 0.01)

old_values = data.drop_duplicates().loc[data['target']==1, 'revenue'].sum()
init_values, _ = get_uplift(data, 1/3, 1/3)


text = """
Без Uplift: {0:.2f}

Результат get_uplift (равные коэффициенты): **{1:.2f}**\n
Результат get_uplift: **{2:.2f}**
"""

if st.button("Рассчитать"):
    result, data = get_uplift(data, a, b)
    data = data[data['rank_pred'] <= 10].groupby(['request_calc_id', 'product_type', 'bank_name'])['rank_pred'].mean().reset_index()
    data = data.groupby(['product_type', 'bank_name'])['rank_pred'].mean().reset_index().sort_values(by='rank_pred', ascending=False)

    data['group'] = data['product_type'] + ' (' + data['bank_name'] + ')'

    st.markdown(text.format(old_values, init_values, result))
    fig = px.bar(data, x='rank_pred', y='group', 
             title="Средняя уверенность модели по типам офферов", 
             labels={'offer_type': 'Тип оффера', 'model_confidence': 'Средняя уверенность модели'}, 
            #  color = 'product_type',
             height=600)
    st.plotly_chart(fig)