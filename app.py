import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



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
data = pd.read_csv('data/20250408_data4streamlit.csv')
grid = pd.read_csv('data/20250408_data43d.csv')


# Интерфейс
st.title("Backtest результативности моделей ранжирования")

st.markdown("Выберите вес для моделей `КК` и `МФО` (вес для модели `МФО` расчитывается как 1 - МФО - КК)")

a = st.slider("Модель КК", 0.0, 1.0, 0.5, 0.01)
b = st.slider("Модель МФО", 0.0, 1.0, 0.5, 0.01)

old_values = data.drop_duplicates().loc[data['target']==1, 'revenue'].sum()
init_values, _ = get_uplift(data, 1/3, 1/3)


text = """
Без Uplift: {0:.2f}

Результат get_uplift (равные коэффициенты): **{1:.2f}**\n
Результат get_uplift: **{2:.2f}**"""

if st.button("Рассчитать"):
    result, data = get_uplift(data, a, b)
    st.markdown(text.format(old_values, init_values, result) + f"({result/old_values:.2f})")

    data_per_product = data[data['rank_pred'] <= 10].groupby(['product_type'])['rank_pred'].mean().reset_index().sort_values(by='rank_pred', ascending=False)
    fig_per_product = px.bar(data_per_product, x='rank_pred', y='product_type', 
             title="Средняя уверенность модели по типам офферов", 
             labels={'offer_type': 'Тип оффера', 'model_confidence': 'Средняя уверенность модели'}, 
             height=600)
    st.plotly_chart(fig_per_product)

    data_per_bank_product = data[data['rank_pred'] <= 10].groupby(['request_calc_id', 'product_type', 'bank_name'])['rank_pred'].mean().reset_index()
    data_per_bank_product = data_per_bank_product.groupby(['product_type', 'bank_name'])['rank_pred'].mean().reset_index().sort_values(by='rank_pred', ascending=False)
    
    data_per_bank_product['group'] = data_per_bank_product['product_type'] + ' (' + data_per_bank_product['bank_name'] + ')'
    data_per_bank_product['group'] = data_per_bank_product['group'].astype('category')
    category_order_group = data_per_bank_product.sort_values(by='rank_pred', ascending=True)['group'].tolist()


    fig_per_bank_product = px.bar(data_per_bank_product,
                                   x='rank_pred', y='group', 
                                   color = 'product_type',
             title="Средняя уверенность модели по типам офферов", 
             labels={'offer_type': 'Тип оффера', 'model_confidence': 'Средняя уверенность модели'},
             orientation='h', 
             height=700,
             category_orders={'group': category_order_group} 
             )
    st.plotly_chart(fig_per_bank_product)


        # Найдём максимум
    max_idx = grid['res'].idxmax()
    max_point = grid.iloc[max_idx]

    # Основной 3D-график
    fig2 = go.Figure(data=[go.Scatter3d(
        x=grid['a'],
        y=grid['b'],
        z=grid['res'],
        mode='markers',
        marker=dict(
            size=4,
            color=grid['res'],
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title='res')
        ),
        name='Точки'
    )])

    # Добавим точку максимума
    fig2.add_trace(go.Scatter3d(
        x=[max_point['a']],
        y=[max_point['b']],
        z=[max_point['res']],
        mode='markers+text',
        marker=dict(
            size=8,
            color='red',
            symbol='diamond'
        ),
        text=[f"MAX: {max_point['res']:.2f}"],
        textposition='top center',
        name='Максимум'
    ))

    # Настройка сцены
    fig2.update_layout(
        scene=dict(
            xaxis_title='a',
            yaxis_title='b',
            zaxis_title='res'
        ),
        title='Интерактивный 3D-график с максимумом',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig2)
