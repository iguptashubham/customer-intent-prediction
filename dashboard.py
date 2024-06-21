import pandas as pd, pickle, streamlit as st, numpy as np
import streamlit.components.v1 as components

st.set_page_config('Customer Intent Predictor',layout='wide')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.title('name Classified')
st.write('organization name is classified')

c1,_,c2 = st.columns([0.5,0.1,0.4])

df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('pipe.pkl','rb'))

with c1:
  st.markdown('### Predictors', unsafe_allow_html=True)
  with st.container(border=True):
    product_cat = st.selectbox(label = 'Select the product category',
                               options = list(df['ProductCategory'].unique()))
    
    product_brand = st.selectbox(label = 'Select the product brand',
                               options = list(df['ProductBrand'].unique()))
    
    price = st.number_input(label='Input the price', value=df['ProductPrice'].quantile(q=0.5))
    
    age = st.number_input(label = 'Enter the age', min_value=df['CustomerAge'].min())
    
    gender = st.selectbox(options=['Male','Female'], label='Select the gender')
    
    if gender == 'Male':
      gender = 1
    else:
      gender = 0
      
    freq = st.number_input(label = 'Enter the frequency', min_value=df['PurchaseFrequency'].min())
    
    satis = st.number_input(label = 'Enter the Satisfaction', min_value=df['CustomerSatisfaction'].min())
    
    q = np.array([[product_cat,product_brand, price, age, gender, freq, satis]])
    pred_data = pd.DataFrame(q, columns=['ProductCategory', 'ProductBrand', 'ProductPrice', 'CustomerAge',
       'CustomerGender', 'PurchaseFrequency', 'CustomerSatisfaction'])
    
  pred = st.button(label = 'Predict')
    
  if pred:
    prediction_result = model.predict(pred_data)
      
    if prediction_result==1:
      st.success('Buying Intent')
    else:
      st.error('Not Buying Intent')
        
with c2:
  st.markdown('#### About')
  st.write(f'[ Name classified ] is an American multinational retail corporation that operates a chain of hypermarkets and stores.')
  
  st.markdown('#### Customer Intent')
  st.write('Customer intent refers to the underlying purpose or goal that drives a customer’s actions or interactions with a business. It’s essential for businesses to understand customer intent to provide personalized and relevant experiences. ')
  
  st.markdown('#### objective')
  st.write(f'Predicting the customer intent of buying the product')
  
  st.markdown('#### Why Important')
  st.write('Predicting customer intent is incredibly valuable for businesses and organizations. Understanding customer intent allows businesses to tailor their interactions, content, and recommendations, resulting in personalized experiences that enhance customer satisfaction and loyalty.')
  
  st.markdown('#### Model')
  st.write('Logistic Regression is a statistical model used for binary classification. It estimates the probability that an event (e.g., a customer making a purchase) occurs based on a set of independent variables . Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities between 0 and 1. The model uses the sigmoid function to map real-valued inputs into this probability range. The goal is to predict whether a customer will take a specific action.🚀')
  
st.markdown('#### Pipeline of Model')
st.markdown('''Certainly! Let's delve into the model pipeline you've created. This pipeline combines several essential steps to build a predictive model:

**Data Preprocessing**:
The pipeline begins by handling raw data. It performs transformations to make the data suitable for modeling. Specifically, it one-hot encodes categorical features (like product category and brand) to convert them into numerical representations. It also standardizes numerical features (such as customer age, purchase frequency, and satisfaction) to have zero mean and unit variance. Additionally, it applies a power transformation (Box-Cox) to the product price feature to improve its distribution.

**Model Selection and Training**:
After preprocessing, the pipeline feeds the transformed data into a logistic regression model. Logistic Regression is a binary classification algorithm that predicts the probability of an event (e.g., customer making a purchase). The model learns from historical data to make predictions based on the input features.

**Hyperparameter Tuning**:
Although not explicitly mentioned, hyperparameters (like solver, maximum iterations, and regularization) can be fine-tuned to optimize model performance. The choice of hyperparameters affects the model's accuracy and generalization ability.

**Model Evaluation**:
It's crucial to assess how well the model performs. Metrics like accuracy, precision, recall, or F1-score help evaluate its effectiveness. Iteratively refine the model based on evaluation results.

**Deployment**:
Once satisfied with the model's performance, deploy it in real-world scenarios (e.g., an application or website). Continuously monitor and maintain the model as new data becomes available.

In summary, this pipeline transforms data, trains a logistic regression model, and prepares it for practical use. 🚀''', unsafe_allow_html=True)
with open('ppeline.html','r') as f: 
    html_data = f.read()
    
components.html(html_data, height = 270)