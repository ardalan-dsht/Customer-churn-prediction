import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle


def user_input():
    st.title("Ardalans churn predictor! :D ")
    st.subheader("This is an online app for our marketing team")
    st.subheader("Whith this app they can predict if a new customer will stay with us or not")
    st.text("Please enter the customers information")
    gender = st.radio("Gender",["Male","Female"])
    credit_score = st.text_input("Enter customers credit score")
    nationality = st.radio("Nationality",["German","French","Spanish"])
    age = st.text_input("How old is the customer?")
    tenure = st.text_input("Whats the customers tenure?")
    balance = st.text_input("Whats the customers bank balance?")
    products = st.text_input("Whats the customers number of products")
    credit = st.radio("Does he/she have a credit card?",["Yes","No"])
    active = st.radio("Is he/she an active member?",["Yes","No"])
    salary = st.text_input("How much is the persons salary?")
    data = {
    "gender" : gender,
    "credit_score" : int(credit_score),
    "nationality" : nationality,
    "age" : int(age),
    "tenure" : int(tenure),
    "balance" : int(balance),
    "products" : int(products),
    "credit" : credit,
    "active" : active,
    "salary" : int(salary)
    }
    dataset = pd.DataFrame(data,index=[0])
dataset = user_input()
file_name = "ardalan_rf_model.pkl"
model = pickle.load( open( file_name, "rb" ) )
prediction = model.predict(dataset)
st.text(prediction)
st.success("Successfully predicted!")
