import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd
import pickle

st.subheader('天津医科大学总医院院内心脏骤停患者死亡风险预测模型')

with st.expander('**相关信息**'):
    st.write('回顾性分析天津市天津医科大学总医院2022年6月—2023年12月发生院内心脏骤停的患者443例。根据排除标准排除92例，最终纳入351例进行研究（图1）。根据最终结局分为存活组131例，死亡组220例。本研究系回顾性分析，符合医学伦理学要求，已获得我院伦理委员会批准（NO:    ）')
    col1, spacer, col2 = st.columns([1,0.1,1])
    with col1:
        st.image('https://p1.ssl.qhimg.com/t0138ab1f7e297446b9.png', width=300)
    with col2:
        st.image('https://p1.ssl.qhimg.com/t0187d84952d73ff63a.jpg', width=300)

with st.form('my_form'):
    st.write('**请按要求输入相关信息，然后按预测键**')

    Age = st.number_input('年龄', min_value=18, max_value=100, step=1)
    Weight = st.number_input('体重', min_value=30, max_value=150, step=1)

    Recovery = st.number_input('复苏时长', min_value=2, max_value=240, step=1)
    Defibrillation = st.number_input('除颤次数', min_value=0, max_value=29, step=1)

    Adrenaline = st.number_input('肾上腺素剂量', min_value=0, max_value=92, step=1)
    Lobeline = st.number_input('洛贝林剂量', min_value=0, max_value=78, step=1)

    submitted = st.form_submit_button('预测')


def norm(x, xmin, xmax):
    x = (x - xmin)/(xmax-xmin)
    return x


with open('C:/Users/AdamN/Desktop/my_model.pkl', 'rb') as file:
    model = pickle.load(file)

with st.expander('**预测结果**'):
    if submitted:
        input_data = pd.DataFrame({'复苏时长': [Recovery],
                                   '年龄': [Age],
                                   '体重': [Weight],
                                   '除颤次数': [Defibrillation],
                                   '肾上腺素剂量': [Adrenaline],
                                   '洛贝林': [Lobeline]})
        input_data['复苏时长'] = norm(input_data['复苏时长'], 2, 240)
        input_data['年龄'] = norm(input_data['年龄'], 16, 100)
        input_data['体重'] = norm(input_data['体重'], 30, 150)
        input_data['除颤次数'] = norm(input_data['除颤次数'], 0, 29)
        input_data['肾上腺素剂量'] = norm(input_data['肾上腺素剂量'], 0, 92)
        input_data['洛贝林'] = norm(input_data['洛贝林'], 0, 78)
        y_pred = model.predict(input_data)
        y_prob = model.predict_proba(input_data)[:, 1]
        y_prob_percentage = f"{y_prob[0] * 100:.2f} %"
        if y_pred[0] == 0:
            st.subheader(f'该患者的死亡风险较小，概率为：{y_prob_percentage}')
        else:
            st.subheader('注意！该患者的死亡风险较大')
            st.subheader(f'预测出的概率为，概率为：{y_prob_percentage}')
    else:
        st.subheader('请完成您的输入！')
