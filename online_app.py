# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:17:58 2020

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
import streamlit as st
import time
import tempfile
import plotly.express as px
from scipy.signal import savgol_filter
import altair as alt
from scipy import spatial
#from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler 
#import matplotlib.pyplot as plt
from scipy.spatial import distance
import collections
from sklearn.metrics import confusion_matrix

st.title('SeaGrassDetect: SEAGRASS DETECTION FROM UNDERWATER VIDEOS')

#path='C:\\Users\\says\\Documents\\Python Scripts\\'
#filename="STACK125_1.mpg"
#filename = '2019_96_STB_NW_Nordskov.mp4'


#img = cv2.imread('C:\\Users\\says\\Documents\\Python Scripts\\STACK125_1_frame_8000.jpg',0)
#img_col = cv2.imread('C:\\Users\\says\\Documents\\Python Scripts\\STACK125_1_frame_8000.jpg',1)
#height,width = img.shape
#rec_img = np.zeros((height,width), np.uint8)
#st.image(img_col,caption="raw data frame", use_column_width=True,channels='BGR')
#lines = lsd(img)
#count=0
#tot_dist=0
#if st.button('Start_lsd'):
#    
#    for i in range(lines.shape[0]):
#        pt1 = ((lines[i, 0]), (lines[i, 1]))
#        pt2 = ((lines[i, 2]), (lines[i, 3]))
#        pt3 = (int(lines[i, 0]), int(lines[i, 1]))
#        pt4 = (int(lines[i, 2]), int(lines[i, 3]))
#        width1 = lines[i, 4]
#        dst = distance.euclidean(pt1,pt2)
#        
#        if -0.5 < pt1[0]-pt2[0] <= 0.5 or -0.5 < pt1[1]-pt2[1] <= 0.5:
#            continue
#        elif abs(pt1[1]-pt2[1])-0.5 <= abs(pt1[0]-pt2[0]) <= abs(pt1[1]-pt2[1])+0.5:
#            continue
#        else:
#            cv2.line(img_col, pt3, pt4, (0, 0, 255), int(np.ceil(width1 / 2)))
#            count=count+1
#            dst = distance.euclidean(pt1,pt2)
#            tot_dist = tot_dist + dst
#    st.image(img_col,caption="lsd data frame", use_column_width=True,channels='BGR')


#st.title("Play Uploaded File")




#video_file = open('C:\\Users\\says\\Documents\\Python Scripts\\test data\\2019_96_STB_NW_Nordskov.mp4.', 'rb')
##video_file = uploaded_file
#video_bytes = video_file.read()
##vid= lsd(video_bytes)
#st.video(video_bytes)
#expander = st.beta_expander("Instructions")
#expander.write("Here is the tool that will change humanity")
#
   
feature=[]

data_load_state = st.text('Loading data...')
f = st.file_uploader("Upload file")
tfile = tempfile.NamedTemporaryFile(delete=False) 
tfile.write(f.read())
st.write("filename:", f.name)
file_name = tfile.name


#st.text([file_name])

imageLocation = st.empty()

#image_placeholder = st.empty()
fpslimit=1
#startTime = time.time()
#st.text([file_name])
video = cv2.VideoCapture(file_name)
#st.text([file_name])
#video = st.file_uploader("Upload a video...", type=["mp4"])
#video = cv2.VideoCapture('C:\\Users\\says\\Documents\\Python Scripts\\test data\\2019_96_STB_NW_Nordskov.mp4.')
data_load_state.text('Loading data...done!')    
while video.isOpened():
    
    success, img1 = video.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #nowTime = time.time()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_col1 = cv2.cvtColor(img1,cv2.IMREAD_COLOR)
    #imageLocation.image(img_col1, use_column_width=True,channels='BGR')
    img_col = cv2.cvtColor(img1,cv2.IMREAD_COLOR)
#        img = cv2.imread('C:\\Users\\says\\Documents\\Python Scripts\\STACK125_1_frame_8000.jpg',0)
#        img_col = cv2.imread('C:\\Users\\says\\Documents\\Python Scripts\\STACK125_1_frame_8000.jpg',1)
    height,width = img.shape
    rec_img = np.zeros((height,width), np.uint8)
    #st.image(img_col,caption="raw data frame", use_column_width=True,channels='BGR')
    #@st.cache
    lines = lsd(img)
    count=0
    tot_dist=0
    for i in range(lines.shape[0]):
        pt1 = ((lines[i, 0]), (lines[i, 1]))
        pt2 = ((lines[i, 2]), (lines[i, 3]))
        pt3 = (int(lines[i, 0]), int(lines[i, 1]))
        pt4 = (int(lines[i, 2]), int(lines[i, 3]))
        width1 = lines[i, 4]
        dst = distance.euclidean(pt1,pt2)
    
        if -0.5 < pt1[0]-pt2[0] <= 0.5 or -0.5 < pt1[1]-pt2[1] <= 0.5:
            continue
        elif abs(pt1[1]-pt2[1])-0.5 <= abs(pt1[0]-pt2[0]) <= abs(pt1[1]-pt2[1])+0.5:
            continue

        else:
            cv2.line(img_col, pt3, pt4, (0, 0, 255), int(np.ceil(width1 / 2)))
            count=count+1
            dst = distance.euclidean(pt1,pt2)
            tot_dist = tot_dist + dst

    #imageLocation.image(img_col1, use_column_width=True,channels='BGR')
    imageLocation.image(img_col, use_column_width=True,channels='BGR')
    #stframe.image(img_col)
    
    feature.append([count,tot_dist])
    #feature2.append(int(tot_dist))
    
    #st.text([count, tot_dist])
    #chart_data = pd.DataFrame([count],columns=['line'])
    #new_rows= np.asarray([count])
    #feature1.add_rows([count])
    #chart = st.line_chart(np.asarray([count])
#        chart.add_rows(np.asarray([[count]]))

    #st.image(img_col,caption="raw data frame", use_column_width=True,channels='BGR')
chart_data = pd.DataFrame(
feature,
columns=['line', 'length'])

#st.title('SeaGrassDetect: SEAGRASS DETECTION FROM UNDERWATER VIDEOS')

poly_order =  1
window_size=25
smooth_lines = savgol_filter(chart_data['line'], window_size, poly_order)
smooth_length = savgol_filter(chart_data['length'], window_size, poly_order)

chart_data['smooth_line'] = smooth_lines
chart_data['smooth_length'] = smooth_length
st.subheader('Extracted Features')
chart_data

st.subheader('Time profile Feaure 1')    
st.line_chart(chart_data[['line','smooth_line']])



st.subheader('Time profile Feaure 2') 
st.line_chart(chart_data[['length','smooth_length']])

st.subheader('Histogram Feaure 1')             
#hist_values_line = np.histogram(
#    chart_data['line'], bins='auto')[0]
#st.bar_chart(hist_values_line)
fig = px.histogram(chart_data, x="line")
fig.show()


st.subheader('Histogram Feaure 2')             
#hist_values_length = np.histogram(
#    chart_data['length'], bins='auto')[0]
#st.bar_chart(hist_values_length)
fig = px.histogram(chart_data, x="length")
fig.show()
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

fig = px.histogram(chart_data, x="line",color="status_combination")
fig.show()
st.subheader('Final Model Predictions')
st.write(chart_data)
if st.button('save dataframe'):
    open('chart_data.txt', 'w').write(chart_data.to_csv())
    
st.write("Sengupta S, Bjarne Ersbøll, and A Stockmarr, 'SeaGrassDetect: A novel method for detection of seagrass from unlabelled under water videos. Ecological Informatics' (2020): 101083.")
#poly_order =  1
#window_size=25
#smooth_lines = savgol_filter(chart_data['line'], window_size, poly_order)
#smooth_length = savgol_filter(chart_data['length'], window_size, poly_order)
#
#chart_data['smooth_line'] = smooth_lines
#chart_data['smooth_length'] = smooth_length
#
#ind=[]
#for i in range(len(chart_data['line'])):
#    ind.append(i)
#    
#chart_data['index'] = ind    
#
#
#
#st.subheader('Extracted Feature Table')
#st.write(chart_data) 
#
#
#
#   
#    
#st.subheader('Time profile Feaure 1')    
#st.line_chart(chart_data[['line','smooth_line']])
#
##st.subheader('Time profile Feaure 1 Smoothed')    
##st.line_chart(smooth_lines)
#
#
#
#st.subheader('Time profile Feaure 2') 
#st.line_chart(chart_data[['length','smooth_length']])
#
##st.subheader('Time profile Feaure 2 Smoothed')    
##st.line_chart(smooth_length)
#
#st.subheader('Histogram Feaure 1')             
#hist_values_line = np.histogram(
#    chart_data['line'], bins='auto')[0]
#st.bar_chart(hist_values_line)
#plt.hist(chart_data['line'], bins = 'auto')
#
#st.subheader('Histogram Feaure 2')             
#hist_values_length = np.histogram(
#    chart_data['length'], bins='auto')[0]
#st.bar_chart(hist_values_length)
##plt.hist(chart_data['length'], bins = 'auto')
##plt.show()
#
#
#st.subheader('Scatter Plot of Features') 
#fig = px.scatter(
#    x=chart_data["line"],
#    y=chart_data["length"],
#)
#fig.update_layout(
#    xaxis_title="Number of Lines",
#    yaxis_title="Total Length of Lines",
#)
#st.write(fig)
#
#
#st.write(chart_data)
#if st.button('save dataframe'):
#    open('chart_data.csv', 'w').write(chart_data.to_csv())
#
#user_input = st.text_input("Select a threshold for Feature 1", 260)




#a = alt.Chart(chart_data).mark_line(opacity=1).encode(
#    x='index', y='line')
#b = alt.Chart(chart_data).mark_line(opacity=0.6).encode(
#    x='index', y='smooth_line')
#c = alt.layer(a, b)
#st.altair_chart(c, use_container_width=True)
#progress_bar = st.sidebar.progress(0)
#status_text = st.sidebar.empty()
#last_rows = np.random.randn(1, 1)
#chart = st.line_chart(last_rows)
#
#for i in range(1, 101):
#    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#    #status_text.text("%i%% Complete" % i)
#    chart.add_rows(new_rows)
#    progress_bar.progress(i)
#    last_rows = new_rows
#
#
#progress_bar.empty()


## Change frame rate
#import time
#
#fpsLimit = 1 # throttle limit
#startTime = time.time()
#cv = cv2.VideoCapture(0)
#While True:
#    frame = cv.read
#    nowTime = time.time()
#    if (int(nowTime - startTime)) > fpsLimit:
#        # do other cv2 stuff....
#        startTime = time.time() # reset time