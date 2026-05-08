import streamlit as st
import requests

st.title("手写数字识别")

API_KEY = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader("请输入识别图片",type = ["png","jpg","jpeg"])

if uploaded_file is not None:
    get_result_key = f"{API_KEY}/get_result"

    st.image(uploaded_file)

    # ✅ 正确发送文件
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)} # request规定，普通表单/json 用json = / data = ,文件用files =

    response = requests.post(get_result_key,files = files)
    data = response.json()

    st.image(f"data:image/png;base64,{data['Img_base64']}")
    st.success(data["result"])
