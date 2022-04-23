#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, LSTM, Bidirectional, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tqdm import tqdm_notebook
from keras.datasets import mnist
from sklearn.utils import shuffle
from keras.utils import np_utils
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


# ### Reading data from prepared dataset 

# In[2]:


data = pd.read_csv('C:/Users/madha/Downloads/Last Stage/dataset_new.csv').astype('float32')


# In[3]:


data


# # Data Splitting 

# ### Data is split into images and respective labels

# In[4]:


X = data.drop('0',axis = 1)
y = data['0']


# In[5]:


print(len(X))


# In[6]:


print(len(y))


# In[7]:


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.14285)


# In[8]:


print(len(train_x))


# In[9]:


print(len(test_x))


# ### Reshaping data to form an image 

# In[10]:


train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))


# In[11]:


print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)


# ### Creating a dictionary to map integer values to specific characters 

# In[12]:


# Dictionary for getting characters from index values...
symdict =  {0:'0',
                    1:'1',
                     2:'2',
                     3:'3',
                     4:'4',
                     5:'5',
                     6:'6',
                     7:'7',
                     8:'8',
                     9:'9',
                     10:'+',
                     11:'-',
                     12:'x',
                     13:'÷',
                     14:'(',
                     15:')',
                     16:'[',
                     17:']',
                     18:'{',
                     19:'}',
                     20:'a',
                     21:'b',
                     22:'c',
                     23:'d', 
                     24:'e',
                     25:'f',
                     26:'g',
                     27:'h',
                     28:'i',
                     29:'j',
                     30:'k',
                     31:'l',
                     32:'m',
                     33:'n',
                     34:'o',
                     35:'p',
                     36:'q',
                     37:'r',
                     38:'s',
                     39:'t',
                     40:'u',
                     41:'v',
                     42:'w',
                     43:'x', 
                     44:'y',
                     45:'z',
                     46:'A',
                     47:'B',
                     48:'C', 
                     49:'D',
                     50:'E',
                     51:'F',
                     52:'G',
                     53:'H',
                     54:'I',
                     55:'J',
                     56:'K',
                     57:'L',
                     58:'M',
                     59:'N',
                     60:'O',
                     61:'P',
                     62:'Q',
                     63:'R',
                     64:'S',
                     65:'T',
                     66:'U',
                     67:'V',
                     68:'W',
                     69:'X',
                     70:'Y',
                     71:'Z',
                     72:'=',
                     73:'≠', 
                     74:'>',
                     75:'<',
                     76:'≥',
                     77:'≤',
                     78:'&',
                     79:'`',
                     80:':',
                     81:',',
                     82:'.',
                     83:'$',
                     84:'!',
                     85:'∃',
                     86:'@',
                     87:'∀',
                     88:'#',
                     89:'in',
                     90:'∞',
                     91:'∫',
                     92:'lim',
                     93:'log', 
                     94:'%',
                     95:'±',
                     96:'π',
                     97:'′',
                     98:'?',
                     99:'""',
                     100:'^',
                     101:'→',
                     102:'/',
                     103:'√',
                     104:'*',
                     105:'∑',
                     106:'~',
                     107:'_',
                     108:'sin',
                     109:'cos',
                     110:'tan',
                     111:'𝛂',
                     112:'β',
                     113:'𝛾', 
                     114:'δ',
                     115:'ε',
                     116:'ζ',
                     117:'𝜂',
                     118:'θ',
                     119:'λ', 
                     120:'μ',
                     121:'𝜈',
                     122:'π',
                     123:'ρ',
                     124:'σ',
                     125:'𝛕',
                     126:'Φ',
                     127:'ψ',
                     128:'ω',
                     129:'Ꭓ',
                     130:'ɩ',
                     131:'Κ',
                     132:'Ο',
                     133:'Ʊ',
                     134:'ξ'
            }


# ### Visualizing character count in the prepared dataset 

# In[13]:


train_yint = np.int0(y)
count = np.zeros(135, dtype='int')
for i in train_yint:
    count[i] +=1

characters = []
for i in symdict.values():
    characters.append(i)

fig, ax = plt.subplots(1,1, figsize=(10,40))
ax.barh(characters, count)

plt.xlabel("Number of elements ")
plt.ylabel("Characters")
plt.grid()
plt.show()


# # Data Shuffling 

# In[14]:


shuff = shuffle(train_x[:100])
fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()

for i in range(9):
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
plt.show()


# # Reshaping Data

# In[15]:


train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
print("New shape of train data: ", train_X.shape)

test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("New shape of train data: ", test_X.shape)


# In[16]:


trainr_y = to_categorical(train_y, num_classes = 135, dtype='int')
print("New shape of train labels: ", trainr_y.shape)

testr_y = to_categorical(test_y, num_classes = 135, dtype='int')
print("New shape of test labels: ", testr_y.shape)


# # CNN Model

# In[17]:


model = Sequential()

# 1st conv block
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
BatchNormalization()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
BatchNormalization()
model.add(MaxPool2D(pool_size=(2, 2), strides=2))


# 2nd conv block
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'))
BatchNormalization()
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'))
BatchNormalization()
model.add(MaxPool2D(pool_size=(2, 2), strides=2))


#3rd conv block
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'same'))
BatchNormalization()
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'same'))
BatchNormalization()
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.summary()

#model.add(Flatten())
    
# LSTM layer
model.add(layers.Reshape((9, 256)))
model.add(layers.LSTM(256, activation="relu", return_sequences=True))
model.summary()

model.add(Flatten())
model.add(Dense(32,activation ="relu"))
model.add(Dense(64,activation ="relu"))
model.add(Dense(135,activation ="softmax"))    


# # Model Compilation

# In[18]:


model.compile(optimizer = Adam(), loss='mean_squared_error', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.14, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


# # Model Fitting

# In[19]:


history = model.fit(train_X, trainr_y, epochs=10, callbacks=[reduce_lr, early_stop],  validation_data = (test_X,testr_y))


# # Model Summary

# In[20]:


model.summary()


# In[21]:


# model.save(r'model_newest.h5')
# Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")


# # Metrics

# In[22]:


print("Validation accuracy :", history.history['val_accuracy'])
print("Training accuracy :", history.history['accuracy'])
print("Validation loss :", history.history['val_loss'])
print("Training loss :", history.history['loss'])


# In[23]:


history.history.keys()


# In[24]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# In[25]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# # Predictions for testing data

# In[26]:


pred = model.predict(test_X[:135])
print(test_X.shape)


# In[27]:


fig, axes = plt.subplots(3,3, figsize=(8,9))
axes = axes.flatten()

for i,ax in enumerate(axes):
    img = np.reshape(test_X[i], (28,28))
    ax.imshow(img, cmap="Greys")
    pred = symdict[np.argmax(testr_y[i])]
    ax.set_title("Prediction: "+pred)
    ax.grid()


# # Predictions for provided image

# In[28]:


image = cv2.imread('im.jpeg')
gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray.copy(), 120, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
print("Image")
plt.imshow(thresh, cmap="gray")
plt.show()
preprocessed_symb = []
for c in cnt:
    x,y,w,h = cv2.boundingRect(c)
    
    # Creating a rectangle around the character in the original image (for displaying the charaters fetched via contours)
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=1)
    
    # Cropping out the char from the image corresponding to the current contours in the for loop
    symb = thresh[y:y+h, x:x+w]
    
    # Resizing that digit to (18, 18)
    resized_symb = cv2.resize(symb, (16,16))
    
    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_symb = np.pad(resized_symb, ((6,6),(6,6)), "constant", constant_values=0)
    
    # Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_symb.append(padded_symb)
    
print("Contoured Image")
plt.imshow(image, cmap="gray")
plt.show()
# print(preprocessed_symb)    
inp = np.array(preprocessed_symb)


# In[29]:


img_pred = []
print("Prediction: ")

for symb in inp:
    img_pred = symdict[np.argmax(model.predict(symb.reshape(1, 28, 28, 1)))]
    print(img_pred, end = '')
    


# In[30]:


img_pred = []
for symb in inp:
    img_pred = symdict[np.argmax(model.predict(symb.reshape(1, 28, 28, 1)))]
    plt.imshow(symb.reshape(28, 28), cmap="gray")
    plt.show()
    print("Prediction: ", img_pred)


# In[ ]:





# In[ ]:




