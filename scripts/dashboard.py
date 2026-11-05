import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from statsmodels.stats.proportion import proportions_ztest

st.set_page_config(page_title="A/B Dashboard", layout="wide")
st.title("A/B тест: Баннер о бесплатной доставке")

tab1, tab2 = st.tabs(["Дизайн эксперимента", " Результаты A/B теста"])

with tab1:
    st.header("Планирование эксперимента")
    st.write("""
    Перед запуском теста мы зафиксировали параметры:
    *   **Базовая конверсия (Группа A):** 10%
    *   **Целевая конверсия (Группа B):** 12% (ожидаемый эффект +2 п.п.)
    *   **Alpha:** 0.05
    *   **Power:** 80%
    """)
    
    st.image('../plots/design.png', caption="Зависимость мощности и MDE от размера выборки")
    
    st.info("""
    **Вывод по дизайну:** 
    Для надежного детектирования эффекта в +2 п.п. с мощностью 80% нам необходимо 
    по 3532 пользователя в каждой группе. Изначально планировали 2000, но расчет MDE показал, 
    что на такой выборке мы поймаем эффект лишь с вероятностью 50%.
    """)

with tab2:
    df = pd.read_csv('../data/gen_data.csv')
    
    a_conv = df[df['group'] == 'group_a']['if_buyer']
    b_conv = df[df['group'] == 'group_b']['if_buyer']
    a_arpu = df[df['group'] == 'group_a']['revenue']
    b_arpu = df[df['group'] == 'group_b']['revenue']
    
    _, p_conv = proportions_ztest([b_conv.sum(), a_conv.sum()], [len(b_conv), len(a_conv)])
    ci_conv = CompareMeans(DescrStatsW(b_conv), DescrStatsW(a_conv)).tconfint_diff(alpha=0.05, usevar='unequal')
    
    _, p_arpu = stats.ttest_ind(b_arpu, a_arpu, equal_var=False)
    ci_arpu = CompareMeans(DescrStatsW(b_arpu), DescrStatsW(a_arpu)).tconfint_diff(alpha=0.05, usevar='unequal')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Получившаяся конверсия")
        st.metric("Группа A", f"{a_conv.mean()*100:.2f}%")
        st.metric("Группа B", f"{b_conv.mean()*100:.2f}%", f"{(b_conv.mean() - a_conv.mean())*100:+.2f} п.п.")
        st.write(f"p-value: {p_conv:.4f}")
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Bar(x=['A', 'B'], y=[a_conv.mean()*100, b_conv.mean()*100], marker_color=['gray', 'blue']))
        fig_conv.add_trace(go.Scatter(x=['A', 'B'], y=[a_conv.mean()*100, b_conv.mean()*100], 
                                      error_y=dict(type='data', array=[0, (ci_conv[1]-ci_conv[0])/2*100], visible=True), 
                                      mode='markers', marker=dict(color='black', size=8)))
        st.plotly_chart(fig_conv, use_container_width=True)
    
    with col2:
        st.subheader("Средняя выручка с пользователя")
        st.metric("Группа A", f"{a_arpu.mean():.2f} руб.")
        st.metric("Группа B", f"{b_arpu.mean():.2f} руб.", f"{b_arpu.mean() - a_arpu.mean():+.2f} руб.")
        st.write(f"p-value: {p_arpu:.4f}")
    
        fig_arpu = go.Figure()
        fig_arpu.add_trace(go.Bar(x=['A', 'B'], y=[a_arpu.mean(), b_arpu.mean()], marker_color=['gray', 'green']))
        fig_arpu.add_trace(go.Scatter(x=['A', 'B'], y=[a_arpu.mean(), b_arpu.mean()], 
                                      error_y=dict(type='data', array=[0, (ci_arpu[1]-ci_arpu[0])/2], visible=True), 
                                      mode='markers', marker=dict(color='black', size=8)))
        st.plotly_chart(fig_arpu, use_container_width=True)
    
    st.success("Вывод: Конверсия статистически значимо выросла. ARPU вырос, но не статистически значимо в связи со своей высокой дисперсией. Эксперимент прошел удачно. Улучшение можно добавлять")