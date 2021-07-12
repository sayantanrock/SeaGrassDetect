# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:45:32 2020

@author: says
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import numpy as np
import os
from pylsd import lsd
import gc
import tempfile
from scipy.signal import savgol_filter
from scipy import spatial
from scipy.spatial import distance
import plotly.express as px

import collections
from sklearn.metrics import confusion_matrix


chart_data = pd.read_csv('C:\\Users\\says\\Documents\\Python Scripts\\result_test_app.txt',  sep="\t",names=["line","length"],index_col=False)
#data.to_excel(".\\output_new.xlsx") 


st.title('SeaGrassDetect: SEAGRASS DETECTION FROM UNDERWATER VIDEOS')

poly_order =  1
window_size=25
smooth_lines = savgol_filter(chart_data['line'], window_size, poly_order)
smooth_length = savgol_filter(chart_data['length'], window_size, poly_order)

chart_data['smooth_line'] = smooth_lines
chart_data['smooth_length'] = smooth_length
st.subheader('Extracted Features')
chart_data

st.subheader('Time profile Feature 1')    
st.line_chart(chart_data[['line','smooth_line']])



st.subheader('Time profile Feature 2') 
st.line_chart(chart_data[['length','smooth_length']])

#st.subheader('Histogram Feaure 1')             
#hist_values_line = np.histogram(
#    chart_data['line'], bins='auto')[0]
#st.bar_chart(hist_values_line)
#fig = px.histogram(chart_data, x="line")
#fig.show()


#st.subheader('Histogram Feaure 2')             
#hist_values_length = np.histogram(
#    chart_data['length'], bins='auto')[0]
#st.bar_chart(hist_values_length)
#fig = px.histogram(chart_data, x="length")
#fig.show()
#plt.hist(chart_data['length'], bins = 'auto')
#plt.show()    

user_input = st.text_input("Select a threshold for Feature 1", 260)
int(user_input)

l1=[]
l2=[]
length_thresh = [3000, 3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,6300,6400,6500,6600,6700,6800,6900,7000]
#length_thresh = [4500,4510,4520,4530,4540,4550,4560,4570,4580,4590,4600,4610,4620,4630,4640,4650,4660,4670,4680,4690,4700]
for leng in length_thresh:
    p=[]
    for i in range(len(smooth_lines)):
        if (chart_data['smooth_line'][i])>int(user_input):
            p.append('PRESENT')
        else:
            p.append('ABSENT')
    p1=[]
    for i in range(len(smooth_length)):
        if (chart_data['smooth_length'][i])>leng:
            p1.append('PRESENT')
        else:
            p1.append('ABSENT')    

    conf=confusion_matrix(p, p1, labels=["PRESENT", "ABSENT"])
    tp=conf[0,0]+conf[1,1]
    pc=float(tp)/float(np.sum(conf))
    l1.append([leng,pc])
    #print(leng, pc)

optimal=pd.DataFrame(l1, columns=['leng_thresh', 'percentage'])


st.subheader('Selecting optimal value of Feature 2 from Feaure 1') 
fig = px.scatter(
    x=optimal["leng_thresh"],
    y=optimal["percentage"],
)
fig.update_layout(
    xaxis_title="Feature 2 threshold",
    yaxis_title="Percentage match of predictions",
)
st.write(fig)

st.write('Optimal feature 2 threshold is', optimal['leng_thresh'][np.argmax(optimal['percentage'])])
feature2_thresh = optimal['leng_thresh'][np.argmax(optimal['percentage'])]

st.subheader('Present/Absent Model')
st.latex(r'''
         
     Present : \lambda\cdot\frac{Feature1}{Threshold1} + (1-\lambda)\cdot\frac{Feature2}{Threshold2} \geq 1
     
     
     ''')

st.latex(r'''
             
     Absent :  \lambda\cdot\frac{Feature1}{Threshold1} + (1-\lambda)\cdot\frac{Feature2}{Threshold2} < 1
     
     ''')    
#Combining features
st.subheader('Finding Optimal lambda')

user_input1 = st.text_input("Select a step size (between 0 and 1) for finding lambda", 0.1)
user_input1

float_range_array = np.arange(0, 1, float(user_input1))

mat= np.zeros((len(float_range_array), len(float_range_array)) )

for i in range(len(float_range_array)):
    a=[]
    a= float_range_array[i]*chart_data['line']/int(user_input) + (1-float_range_array[i])*chart_data['length']/feature2_thresh
    a[a>1]=1
    a[a<1] =0
    for j in range(i+1, len(float_range_array)):
        b=[]
        b= float_range_array[j]*chart_data['line']/int(user_input) + (1-float_range_array[j])*chart_data['length']/feature2_thresh
        b[b>1]=1
        b[b<1] =0
        mat[i][j] = np.sum(a!=b)
        mat[j][i] = mat[i][j]

minimum=1000000     
for i in range(len(float_range_array)):
    if(i==len(float_range_array)-1):
        break
    else:
         if (mat[i][i+1]< minimum):
             minimum = mat[i][i+1]
             lamda = (i+1)*float(1)/len(float_range_array)
             
        
        
st.subheader('Choose the lamda for minimum mismatch neighbourhood ')
st.write(mat)

st.write('Optimal lambda is', lamda)

st.write('The number of mismatched frames are', minimum)




    
a= lamda*chart_data['line']/int(user_input) + (1-lamda)*chart_data['length']/feature2_thresh
a[a>1]=1
a[a<1] =0
chart_data['status_combination'] = a
    
a= chart_data['line']/int(user_input)
a[a>1]=1
a[a<1] =0
chart_data['status_line'] = a

a= chart_data['length']/feature2_thresh
a[a>1]=1
a[a<1] =0
chart_data['status_length'] = a

st.subheader('Prediction using only Feature 1') 
fig = px.scatter(
    x=chart_data["line"],
    y=chart_data["length"],
    color=chart_data["status_line"]
)
fig.update_layout(
    xaxis_title="Number of Lines",
    yaxis_title="Total Length of Lines",
)
st.write(fig)

st.subheader('Prediction using only Feature 2') 
fig = px.scatter(
    x=chart_data["line"],
    y=chart_data["length"],
    color=chart_data["status_length"]
)
fig.update_layout(
    xaxis_title="Number of Lines",
    yaxis_title="Total Length of Lines",
)
st.write(fig)




st.subheader('Prediction using Combination method') 
fig = px.scatter(
    x=chart_data["line"],
    y=chart_data["length"],
    color=chart_data["status_combination"]
)
fig.update_layout(
    xaxis_title="Number of Lines",
    yaxis_title="Total Length of Lines",
)
st.write(fig)

#df = px.data.tips()

#fig = px.histogram(chart_data, x="line",color="status_combination")
#fig.show()
st.subheader('Final Model Predictions')
st.write(chart_data)
if st.button('save dataframe'):
    open('chart_data.txt', 'w').write(chart_data.to_csv())
    
st.write("Sengupta S, Bjarne Ersbøll, and A Stockmarr, 'SeaGrassDetect: A novel method for detection of seagrass from unlabelled under water videos. Ecological Informatics' (2020): 101083.")