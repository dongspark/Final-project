import spacy
spacy.cli.download("en_core_web_md")
import random
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
import pickle
from tensorflow import keras
import altair as alt
import numpy as np 
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


import torch 
import IPython.display as ipd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.models import vgg16
from tqdm.notebook import tqdm
from CLIP import clip # The clip model
from torchvision import transforms # Some useful image transforms
import torch.nn.functional as F # Some extra methods we might need
from tqdm.notebook import tqdm
import IPython.display as ipd
from IPython.display import display


from omegaconf import OmegaConf
import sys
sys.path.append('./taming_transformers')
from taming_transformers.taming.models import cond_transformer, vqgan
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_poem(input_word):
        
    #load all the data you need:
    model = keras.models.load_model('models/bd_bi_lstm')
    word_index = pickle.load(open('bd_word_index.p','rb'))
    input_sequences = pickle.load(open('bd_input_sequences.p','rb'))
    max_sequence_len = max([len(x) for x in input_sequences])
    corpus_cleaned = pd.read_csv('bd_corpus_cleaned.csv')
    tokenizer = pickle.load(open('bd_tokenizer.p','rb'))

        
    
    #input_word = input("Enter a word: ")
    nlp = spacy.load('en_core_web_md')
    topic = nlp(input_word)
    
    #Create a list for the similarities:
    sim_list = []
    rand_word_list=[]
    
    #choose 10 or so random words from word_index of data:
    for i in range(0, 20):
        rand_word = random.choice(list(word_index.keys()))
        #find word in spacy
        rand_word_spacy = nlp(rand_word)
        rand_word_list.append(rand_word)
    
        #Compute similarity for the word:
        similarity = (topic.similarity(rand_word_spacy))
        sim_list.append(similarity)
        
    #saves similarity to DataFrame
    df_1 = pd.DataFrame({'similarity' : sim_list, 'word' : rand_word_list })
    df_1.sort_values(by='similarity', inplace=True, ascending=False)
    
    
    #create input words
    words_to_use=[input_word] #'the ' + 
    
    for i in range(0,2):
        words_to_use.append(df_1['word'][i])
        
    
        
    #Create the actual poem:
    poem = []
    
    choices = random.sample(words_to_use, 3)
    for item in choices: 
        seed_text = item
        next_words = random.choice([6,7,8,9,10,11,12])
        
        
    
        output_index = [0,0,0]
        output_word = [0,0,0]
        
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    
            
            # if-clauses I added, because otherwise the model sometimes gives the same word several times in a row:
            if predicted[0] == output_index[-1]:
                predicted = np.argsort(model.predict(token_list), axis=-1)[0][-2]
            
            # also this. Otherwise it sometimes gives a pair of two words several times in a row:
            if (predicted[0] == output_index[-2]) & (output_word[-1] == output_word[-3]):
                predicted = np.argsort(model_mo.predict(token_list), axis=-1)[0][-2]
            
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word.append(word)
                    output_index.append(index)
                    break
            seed_text += " " + output_word[-1]
        
        poem.append(seed_text)
    return poem[0]+'\n'+ poem[1]+'\n'+ poem[2]




def get_image(input_word):    


        def load_vqgan_model(config_path, checkpoint_path):
            config = OmegaConf.load(config_path)
            if config.model.target == 'taming.models.vqgan.VQModel':
                model = vqgan.VQModel(**config.model.params)
                model.eval().requires_grad_(False)
                model.init_from_ckpt(checkpoint_path)
            elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
                parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
                parent_model.eval().requires_grad_(False)
                parent_model.init_from_ckpt(checkpoint_path)
                model = parent_model.first_stage_model
            else:
                raise ValueError(f'unknown model type: {config.model.target}')
            del model.loss
            return model

        class ReplaceGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_forward, x_backward):
                ctx.shape = x_backward.shape
                return x_forward
         
            @staticmethod
            def backward(ctx, grad_in):
                return None, grad_in.sum_to_size(ctx.shape)
         
         
        replace_grad = ReplaceGrad.apply
        
         
        class ClampWithGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, min, max):
                ctx.min = min
                ctx.max = max
                ctx.save_for_backward(input)
                return input.clamp(min, max)
         
            @staticmethod
            def backward(ctx, grad_in):
                input, = ctx.saved_tensors
                return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
         
         
        clamp_with_grad = ClampWithGrad.apply
        


        def vector_quantize(x, codebook):
          d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
          indices = d.argmin(-1)
          x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
          return replace_grad(x_q, x)

        def synth(z):
          z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
          return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

        def rand_z(width, height):
          f = 2**(model.decoder.num_resolutions - 1)
          toksX, toksY = width // f, height // f
          n_toks = model.quantize.n_e
          one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
          z = one_hot @ model.quantize.embedding.weight
          z = z.view([-1, toksY, toksX, model.quantize.e_dim]).permute(0, 3, 1, 2)
          return z
        model = load_vqgan_model('vqgan_imagenet_f16_16384.yaml', 'vqgan_imagenet_f16_16384.ckpt').to(device)
        perceptor = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
   
        def clip_loss(im_embed, text_embed):
          im_normed = F.normalize(im_embed.unsqueeze(1), dim=2)
          text_normed = F.normalize(text_embed.unsqueeze(0), dim=2)
          dists = im_normed.sub(text_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2) # Squared Great Circle Distance
          return dists.mean()

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711])



        
        prompt_text =input_word#@param
        width = width_input #@param
        height = hight_input #@param
        lr = lr_input #@param
        n_iter = n_iter_input #@param
        crops_per_iteration = 8 #@param

        # The transforms to get variations of our image
        tfms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomAffine(5),
            transforms.ColorJitter(),
            transforms.GaussianBlur(5),
        ])

        # The z we'll be optimizing
        z = rand_z(width, height)
        z.requires_grad=True
        # The text target
        text_embed = perceptor.encode_text(clip.tokenize(prompt_text).to(device)).float()

        # The optimizer - feel free to try different ones here
        optimizer = torch.optim.Adam([z], lr=lr, weight_decay=1e-6)

        losses = [] # Keep track of our losses (RMSE values)

        # A folder to save results
        #!rm -r steps
        #!mkdir steps


        # Display for showing progress
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        p = display(fig, display_id=True) 

        # The optimization loop:
        for i in tqdm(range(n_iter)):
          # Reset everything related to gradient calculations
          optimizer.zero_grad()

          # Get the GAN output
          output = synth(z)

          # Calculate our loss across several different random crops/transforms
          loss = 0
          for _ in range(crops_per_iteration):
            image_embed = perceptor.encode_image(tfms(normalize(output)).to(device)).float()
            loss += clip_loss(image_embed, text_embed)/crops_per_iteration

          # Store loss
          losses.append(loss.detach().item())

          # Save image
          im_arr = np.array(output.cpu().squeeze().detach().permute(1, 2, 0)*255).astype(np.uint8)
          #Image.fromarray(im_arr).save(f'steps/{i:04}.jpeg')
        
          # Update plots 
          #if i % 5 == 0: # Saving time
          #  axs[0].plot(losses)
          #  axs[1].imshow(im_arr)
          #  p.update(fig)
          # Backpropagate the loss and use it to update the parameters
          loss.backward() # This does all the gradient calculations
          optimizer.step() # The optimizer does the update
          #ipd.clear_output()
        return im_arr



st.image('titlepic.png',width=800)


st.write("""
# Lyrics and image Web App
This app generate Bob Dylan's style of lyrics and customized image according to your input! :memo: :frame_with_picture:
***
""")

seed_input='the blue sky'


st.sidebar.image("sidepic.png", width=300)

st.sidebar.header('User Input Features')
seed=st.sidebar.text_input('Seed words input',seed_input)


st.sidebar.subheader(':point_right: Select the width of the image')
width_input=st.sidebar.slider('choose between 0-1000',min_value=0,max_value=1000,step=25,value=40)


st.sidebar.subheader(':point_right:Select the height of the image')
hight_input=st.sidebar.slider('choose between 0-750',min_value=0,max_value=750,step=5,value=30)

st.sidebar.subheader(':point_right:Select the learning rate of the image generator')
lr_input=st.sidebar.slider('choose between 0.00-1.00',min_value=0.00,max_value=1.00,step=0.01,value=0.1)

st.sidebar.subheader(':point_right:Select the number of literation for the image generator')
n_iter_input=st.sidebar.slider('choose between 50-1500',min_value=50,max_value=1500,step=50,value=100)


st.sidebar.subheader(':point_right:Select the style of the image')
style = st.sidebar.selectbox('Select the style of the image',
    ('pencil sketch', 'anime', 'watercolor','fine art','normal'))


st.header('Generated lyrics :memo:')

if st.sidebar.button('Start to generate'):
    x=get_poem(seed)
    x = x.split('\n')
    x
    st.header('Generated image :frame_with_picture:')
    x = ''.join(x) # Concatenates list to string
    #z=style +' painting of '+ x
    y=get_image(style +' painting of '+ x)

    st.image(y,width=400)
