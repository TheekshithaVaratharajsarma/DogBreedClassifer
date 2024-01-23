# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:53:50 2024

@author: user
"""

import os
import base64
from io import BytesIO
import requests
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import dash
from dash import Dash, dcc, html, Input, Output, State
from dash.dependencies import MATCH, ALL
from keras.preprocessing import image
from keras.models import load_model
import dash_bootstrap_components as dbc

model_url = "https://github.com/TheekshithaVaratharajsarma/DogBreedClassifer/releases/tag/v1.0.0"
model_response = requests.get(model_url)

if model_response.status_code == 200:
    model = load_model(BytesIO(model_response.content))
else:
    raise Exception(f"Failed to download the model. Status code: {model_response.status_code}")

with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Convert the background image to base64
with open('background.jpg', 'rb') as image_file:
    base64_encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
app = Flask(__name__)
server = app.server

dash_app = Dash(
    __name__, 
    server=app, 
    url_base_pathname='/dashboard/', 
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

dash_app.layout = html.Div([
    html.H1("Dog Breed Identifier", className='header', style={'text-align': 'center'}),
    
    html.Div([
        dbc.Button(
            "Let's Start!", 
            id='open-container-btn', 
            style={
                'backgroundColor': '#ff6600', 
                'border-color': 'white'
            }
        ),
    ], style={'text-align': 'center', 'margin-top': '20px'},
        id='start-button-container'),

    html.Div([
        html.Div([
            html.Div([
                dcc.Input(id='image-url-input', type='text', placeholder='Enter Image URL'),
                html.Button(
                    'Submit', 
                    id='submit-url-button', 
                    className='btn btn-primary',   
                    style={
                        'backgroundColor': '#ff6600', 
                        'border-color': 'white'
                    }
                ),
            ],  
                id='url-container',
                style={
                    'display': 'flex',
                    'margin': '10px',
                    'justify-content': 'space-around',
                    'align-items' : 'center'
                }
            ),

            html.Div(id='output-container', className='output-container', style={'display': 'none'}),

            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=[
                        html.Div([
                            'Drag and Drop',
                            html.Br(),
                            'or',
                            html.Br(),
                            html.A('Click to Select an Image')
                        ], style={'color': 'white'}),
                    ],
                    multiple=False,
                ),
            ], 
                className='drag-drop-section',
                style={
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'height': '90%',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '15px',
                    'background-color': '#cccccc',
                    'margin': '20px 30px',
                },
                id='drag-drop-container'
            ),

            html.Div([
                html.Img(
                    key='image-key',
                    id='uploaded-image', 
                    className='uploaded-image', 
                    style={
                        'height': '80%',
                        'width': '80%'
                    }
                ),
                html.Button(
                    'Delete Image',
                    id='delete-image-button',
                    style={'display': 'none', 'backgroundColor': '#ff6600', 'border-color': 'white'},
                    className='btn btn-danger',
                ),
            ], 
                id='result-container', 
                style={
                    'display': 'none',
                    'height': '90%',
                    'flex-direction': 'column',
                    'align-items': 'center',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '15px',
                    'background-color': '#cccccc',
                    'margin': '20px 30px',
                }
            ),

            html.Div([
                html.Button(
                    'Predict', 
                    id='predict-button', 
                    className='btn btn-primary', 
                    style={
                        'backgroundColor': '#ff6600', 
                        'border-color': 'white'
                    }
                ),
            ], style={'display': 'none', 'margin': '10px'},
                id='predict-button-container',
            ),

            html.Div([
                html.Button(
                    'Upload Another Image', 
                    id='upload-another-button', 
                    className='btn btn-primary', 
                    style={
                        'backgroundColor': '#ff6600', 
                        'border-color': 'white',
                    }
                ),
            ], id='upload-another-container', style={'display': 'none', 'margin': '10px'}),
        ], 
            className='upload-container', 
            style={
                'display': 'flex',
                'flex-direction': 'column',
                'justify-content': 'flex-start', 
                'height': '100%',
                'width': '30vw',
                'min-width': '272px',
                'background-color': 'white',
                'borderWidth': '5px',
                'borderRadius': '15px',
            }
        ),
    ], 
        id='container',
        style={
            'text-align': 'center', 
            'margin-top': '40px', 
            'height': '60%', 
            'display': 'flex', 
            'justify-content': 'center',
        },
        className='container-section',
    ),

    html.Div([
        dcc.Loading(
            id="loading",
            type="circle",
            fullscreen=False,
            style={'zIndex': '1000'},
        ),
    ], id='loading-container'),
    
    html.Div([
        html.P(
            '© Theekshitha. CNN model evaluated results - Only for academic purposes. All rights reserved.', 
            style={
                'color': '#333333', 
                'font-size': '14px', 
                'margin-top': '20px', 
                'position': 'fixed',
                'left': '0',
                'bottom': '0',
                'width': '100%',
                'text-align': 'center'
            }
        ),
    ], id='footer-container', style={'text-align': 'center'}),
    dcc.Store(id='global-store'),
], style={'background-image': f'url("data:image/jpg;base64,{base64_encoded_string}")', 'background-size': 'cover', 'height': '100vh'})

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_breed(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    breed = labels[predicted_class]
    return breed

# Callbacks to control visibility and behavior
@dash_app.callback(
    [Output('container', 'style'),
     Output('start-button-container', 'style')],
    [Input('open-container-btn', 'n_clicks')],
)
def show_container(n_clicks):
    if n_clicks is None:
        return {'display': 'none'}, {'text-align': 'center', 'margin-top': '20px'}
    else:
        return {'text-align': 'center', 'margin-top': '40px', 'height': '60%', 'display': 'flex', 'justify-content': 'center'}, {'display': 'none'}

# Define callback to handle image upload 
@dash_app.callback(
    [Output('uploaded-image', 'src'),
     Output('drag-drop-container', 'style'),
     Output('result-container', 'style'),
     Output('predict-button-container', 'style'),
     Output('output-container', 'style'),
     Output('delete-image-button', 'style'),
     Output('output-container', 'children'),
     Output('url-container', 'style'),
     Output('upload-another-container', 'style'),
     Output('image-url-input', 'value')],
    [Input('submit-url-button', 'n_clicks'),
     Input('delete-image-button', 'n_clicks'),
     Input('predict-button', 'n_clicks'),
     Input('upload-another-button', 'n_clicks'),
     Input('upload-image', 'contents')],
    [State('image-url-input', 'value'),
     State('uploaded-image', 'src')],
    prevent_initial_call=True
)
def update_image(n_clicks_submit, n_clicks_delete, n_clicks_predict, n_clicks_upload_another, contents, image_url, img):
    ctx = dash.callback_context

    if ctx.triggered_id == 'submit-url-button' and image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_content = response.content
                encoded_image = base64.b64encode(image_content).decode()
                img_src = f'data:image/png;base64,{encoded_image}'
                return img_src, {'display': 'none'}, {
                        'display': 'flex', 
                        'flex-direction': 'column',
                        'align-items': 'center',
                        'height': '90%',
                        'max-height': '320px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '15px',
                        'margin': '20px 30px'
                    }, {'display': 'block', 'margin': '10px'}, {'display': 'none'}, {
                            'display': 'block',
                            'backgroundColor': '#ff6600', 
                            'border-color': 'white'
                        }, '', {
                                'display': 'flex',
                                'margin': '10px',
                                'justify-content': 'space-around',
                                'align-items' : 'center'
                            }, {'display': 'none'}, image_url
        except Exception as e:
            print(f"Error fetching image from URL: {e}")

    elif ctx.triggered_id == 'upload-image' and contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            img_src = f'data:image/png;base64,{base64.b64encode(decoded).decode()}'
            return img_src, {'display': 'none'}, {
                        'display': 'flex',
                        'flex-direction': 'column',
                        'align-items': 'center', 
                        'height': '90%',
                        'max-height': '320px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '15px',
                        'margin': '20px 30px'
                    }, {'display': 'block', 'margin': '10px'}, {'display': 'none'}, {
                            'display': 'block',
                            'backgroundColor': '#ff6600', 
                            'border-color': 'white'
                        }, '', {
                                'display': 'flex',
                                'margin': '10px',
                                'justify-content': 'space-around',
                                'align-items' : 'center'
                            }, {'display': 'none'}, ''

    elif ctx.triggered_id == 'delete-image-button':
        return '', {'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'height': '90%',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '15px',
                    'background-color': '#cccccc',
                    'margin': '20px 30px'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, '', {
                        'display': 'flex',
                        'margin': '10px',
                        'justify-content': 'space-around',
                        'align-items' : 'center'
                    }, {'display': 'none'}, ''
    elif ctx.triggered_id == 'predict-button' and img:
        img_type, img_string = img.split(',')
        if 'image' in img_type:
            # Decode the base64 image
            decoded = Image.open(BytesIO(base64.b64decode(img_string)))
            # Save the image locally
            img_path = 'uploaded_image.jpg'
            decoded.save(img_path)
            # Preprocess the image
            img_array = preprocess_image(img_path)
            # Make predictions
            breed = predict_breed(img_array)

            # Display the breed name
            output = f'The predicted dog breed is: {breed}'
            return img, {'display': 'none'}, {
                            'display': 'flex',
                            'flex-direction': 'column',
                            'align-items': 'center',
                            'justify-content': 'center',
                            'height': '90%',
                            'max-height': '320px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '15px',
                            'margin': '20px 30px'
                        }, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, output, {'display': 'none'}, {
                                'display': 'block', 'margin': '10px'}, ''
    
    elif ctx.triggered_id == 'upload-another-button':
        return '', {'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'height': '90%',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '15px',
                    'background-color': '#cccccc',
                    'margin': '20px 30px'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, '', {
                        'display': 'flex',
                        'margin': '10px',
                        'justify-content': 'space-around',
                        'align-items' : 'center'
                    }, {'display': 'none'}, ''
    else:
        return '', {'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'height': '90%',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '15px',
                    'background-color': '#cccccc',
                    'margin': '20px 30px'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, '', {
                        'display': 'flex',
                        'margin': '10px',
                        'justify-content': 'space-around',
                        'align-items' : 'center'
                    }, {'display': 'none'}, ''

# Serve bg image
@app.route('/background.jpg')
def bg_image():
    return send_from_directory('.', 'background.jpg')

# Serve uploaded images
@app.route('/uploaded_image.jpg')
def uploaded_image():
    return send_from_directory('.', 'uploaded_image.jpg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



