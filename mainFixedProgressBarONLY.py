# Import the libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib

matplotlib.use('Agg')


def data1():  # this function load data file
    import pandas as pd
    import numpy as np
    # Load the training data
    data = pd.read_csv("data/train.csv")
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    return data


def cleanText(text):  # this function clean text
    text = BeautifulSoup(text, "lxml").text
    text = text.replace('\n', '')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


def tfidt(text_matrix):  # this function take batch and apply TF-IDF
    count_vect = CountVectorizer(max_features=100)  # We are extracting only 100 features for our text.
    X_train_counts = count_vect.fit_transform(text_matrix)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf = X_train_tfidf.toarray()
    return X_train_tfidf


def applyAttention(wordVectorRowsInSentence):  # this function take TF-IDF matrix and Apply Attention
    N_f = wordVectorRowsInSentence.shape[-1]
    uiVectorRowsInSentence = tf.keras.layers.Dense(units=100, activation='tanh')(
        wordVectorRowsInSentence)  # [*, N_s, N_a]
    vVectorColumnMatrix = tf.keras.layers.Dense(units=1, activation='tanh')(uiVectorRowsInSentence)  # [*, N_s, 1]
    vVector = tf.keras.layers.Flatten()(vVectorColumnMatrix)  # [*, N_s]
    attentionWeightsVector = tf.keras.layers.Activation('softmax', name='attention_vector_layer')(vVector)  # [*,N_s]
    attentionWeightsMatrix = tf.keras.layers.RepeatVector(N_f)(attentionWeightsVector)  # [*,N_f, N_s]
    attentionWeightRowsInSentence = tf.keras.layers.Permute([2, 1])(attentionWeightsMatrix)  # [*,N_s, N_f]
    attentionWeightedSequenceVectors = tf.keras.layers.Multiply()(
        [wordVectorRowsInSentence, attentionWeightRowsInSentence])  # [*,N_s, N_f]
    attentionWeightedSentenceVector = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1),
                                                             output_shape=lambda s: (s[0], s[2]))(
        attentionWeightedSequenceVectors)  # [*,N_f]
    return attentionWeightedSentenceVector


def makeBatch(data):
    data_title = data['title'].to_numpy()
    data_title = [cleanText(i) for i in data_title]  # apply cleantext function on data title
    data_title = np.array(data_title)
    data_title = data_title[:18000]
    data_title = data_title.reshape(18, 1000)  # make 18 batch
    # apply tf-idf on every batch
    X_train_tfidf_1 = tfidt(data_title[0])
    X_train_tfidf_2 = tfidt(data_title[1])
    X_train_tfidf_3 = tfidt(data_title[2])
    X_train_tfidf_4 = tfidt(data_title[3])
    X_train_tfidf_5 = tfidt(data_title[4])
    X_train_tfidf_6 = tfidt(data_title[5])
    X_train_tfidf_7 = tfidt(data_title[6])
    X_train_tfidf_8 = tfidt(data_title[7])
    X_train_tfidf_9 = tfidt(data_title[8])
    X_train_tfidf_10 = tfidt(data_title[9])
    X_train_tfidf_11 = tfidt(data_title[10])
    X_train_tfidf_12 = tfidt(data_title[11])
    X_train_tfidf_13 = tfidt(data_title[12])
    X_train_tfidf_14 = tfidt(data_title[13])
    X_train_tfidf_15 = tfidt(data_title[14])
    X_train_tfidf_16 = tfidt(data_title[15])
    X_train_tfidf_17 = tfidt(data_title[16])
    X_train_tfidf_18 = tfidt(data_title[17])

    # apply Attention on every IF-IDF matrix
    X_train_tfidf_1 = np.array(applyAttention(X_train_tfidf_1))
    X_train_tfidf_2 = np.array(applyAttention(X_train_tfidf_2))
    X_train_tfidf_3 = np.array(applyAttention(X_train_tfidf_3))
    X_train_tfidf_4 = np.array(applyAttention(X_train_tfidf_4))
    X_train_tfidf_5 = np.array(applyAttention(X_train_tfidf_5))
    X_train_tfidf_6 = np.array(applyAttention(X_train_tfidf_6))
    X_train_tfidf_7 = np.array(applyAttention(X_train_tfidf_7))
    X_train_tfidf_8 = np.array(applyAttention(X_train_tfidf_8))
    X_train_tfidf_9 = np.array(applyAttention(X_train_tfidf_9))
    X_train_tfidf_10 = np.array(applyAttention(X_train_tfidf_10))
    X_train_tfidf_11 = np.array(applyAttention(X_train_tfidf_11))
    X_train_tfidf_12 = np.array(applyAttention(X_train_tfidf_12))
    X_train_tfidf_13 = np.array(applyAttention(X_train_tfidf_13))
    X_train_tfidf_14 = np.array(applyAttention(X_train_tfidf_14))
    X_train_tfidf_15 = np.array(applyAttention(X_train_tfidf_15))
    X_train_tfidf_16 = np.array(applyAttention(X_train_tfidf_16))
    X_train_tfidf_17 = np.array(applyAttention(X_train_tfidf_17))
    X_train_tfidf_18 = np.array(applyAttention(X_train_tfidf_18))

    # use 15 batch for training and 3 for test
    x_train_2d = np.concatenate((X_train_tfidf_1, X_train_tfidf_2, X_train_tfidf_3, X_train_tfidf_4, X_train_tfidf_5,
                                 X_train_tfidf_6, X_train_tfidf_7, X_train_tfidf_8, X_train_tfidf_9, X_train_tfidf_10,
                                 X_train_tfidf_11, X_train_tfidf_12, X_train_tfidf_13, X_train_tfidf_14,
                                 X_train_tfidf_15), axis=0)
    x_test_2d = np.concatenate((X_train_tfidf_16, X_train_tfidf_17, X_train_tfidf_18), axis=0)

    # convert 2d array to 3d array because lstm take 3d array as input
    x_train = x_train_2d[:, None]
    x_test = x_test_2d[:, None]

    # take label train and test from data
    data_label = data['label'].to_numpy()
    data_label = data_label[:18000]

    y_train = data_label[:15000]
    y_test = data_label[15000:]

    return x_train, x_test, y_train, y_test


def model_lstm():
    # Creating model
    data_dim = 100
    model = Sequential()
    model.add(LSTM(1024, input_shape=(None, data_dim), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def earlyStoppingAndTrainModel(x_train, x_test, y_train, y_test):
    # early stopping
    early_stop = EarlyStopping(monitor='val_loss', mode='min',
                               verbose=1, patience=5)
    # check best model and save it
    checkpoint = ModelCheckpoint('modelCallRumor.h5', monitor='val_accuracy',
                                 verbose=1, save_best_only=True, mode='max')
    model = model_lstm()
    start = datetime.now()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1000,
                        callbacks=[early_stop, checkpoint])
    time = datetime.now() - start
    return history, time, start


def History(history, time, start):
    # Displaying curves of loss and accuracy during training
    import matplotlib.pyplot as plt
    fig = plt.figure()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    no_of_epochs = len(acc)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.close()
    plt.savefig('pics/graph_1.png')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    x = plt.show()
    plt.savefig('pics/graph_2.png')
    acc = val_acc[-1]
    los = val_loss[-1]
    acc = "{:.2f}".format(acc)
    los = "{:.2f}".format(los)
    return plt, time, start, no_of_epochs, acc, los


def main_train():
    data = data1()
    x_train, x_test, y_train, y_test = makeBatch(data)
    history, time, start = earlyStoppingAndTrainModel(x_train, x_test, y_train, y_test)
    plt, time, start, no_of_epochs, acc, los = History(history, time, start)
    return plt, time, start, no_of_epochs, acc, los


#prdicte code start here
def data_pre(txt):#this function load data file
    # Load the training data
    data = pd.read_csv("data/train.csv")
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    f = open(txt, "r")
    txt = f.read()
    df2 = pd.DataFrame([[0,txt,'txt','txt',0]],columns=['id','title','author','text','label'])
    data = data.append(df2, ignore_index=True)
    return data


def makeTestMatrix(data):
    data_title = data['title'].to_numpy()
    data_title = [cleanText(i) for i in data_title]  # apply cleantext function on data title
    data_title = np.array(data_title)
    data_title = data_title[17500:]  # some part of data include txt file
    # apply tf-idf
    X_train_tfidf_1 = tfidt(data_title)

    # apply Attention on  IF-IDF matrix
    X_train_tfidf_1 = np.array(applyAttention(X_train_tfidf_1))

    x_train_2d = X_train_tfidf_1

    # convert 2d array to 3d array because lstm take 3d array as input
    x_train_pre = x_train_2d[:, None]

    return x_train_pre

def predict_1(x_train):# load model h5 file and predict model
    from tensorflow.keras.models import load_model
    model = load_model('modelCallRumor.h5')
    y = -1
    predict = model.predict(x_train[y,None])
    predict_index = np.argmax(predict[0])
    if predict_index==0:
        img = 'pics/not_rumor.png'
        result = str(int(predict[0][0]*100))+"% Non-Rumor\n"+str(int((1-predict[0][0])*100))+"% Rumor"
    else:
        img='pics/rumor.png'
        result = str(int((1-predict[0][1])*100))+"% Non-Rumor\n"+str(int(predict[0][1]*100))+"% Rumor"
    return result , predict_index,img

def main_predict(txt):
    data = data_pre(txt)
    x_train_pre = makeTestMatrix(data)
    result , predict_index,img = predict_1(x_train_pre)
    return result , predict_index,img
#All the imports here
try:
    import Tkinter as tk
    import tkFont
    import ttk
    from tkinter.filedialog import askopenfilename



except ImportError:  # Python 3
    import tkinter as tk
    from tkinter import *
    import tkinter.font as tkFont
    import tkinter.ttk as ttk


# !!!!!!!!!!!!!!!!Main Page !!!!!!!!!!!!!!

class ThisPage():
    def __init__(self,master):
        print("in this page")
        # initializing all necessary variables
        self.root = master
        self.root.title("Main Page")
        # Some Usefull variables
        self.username = StringVar()
        self.password = StringVar()
        self.type = IntVar()
        self.canvas_width = self.root.winfo_screenwidth()
        self.canvas_height = self.root.winfo_screenheight()
        self.python_green = "white"
        # Create Widgets function
        self.widgets()
# All the widgets for Main Page
    def widgets(self):
        print("in widgets")
        self.logf = Canvas(self.root, bg='slateblue', height=self.canvas_height, width=self.canvas_width)

        points = [self.canvas_width, 450, self.canvas_width, self.canvas_height, 1000, self.canvas_height]
        self.logf.create_polygon(points, outline=self.python_green,
                                 fill='white', width=3)
        width=self.canvas_width/2-270
        Label(self.logf, text='CALL AT RUMOR ', fg="white", bg='slateblue', font=('arial 50 bold '), pady=5, padx=5) \
            .place(x=width, y=160)

        # Buttons with their actions written in command
        self.b3 = Button(self.logf, text="<< RUMOR?", fg="white", font=('arial 30 bold'), bg='slateblue', borderwidth=0,
                         command=self.go_to_Rumor )
        self.b3.place(x=width-110, y=380)

        self.b4 = Button(self.logf, text="TRAIN!>>", fg="white", font=('arial 30 bold'), bg='slateblue', borderwidth=0,
                         command=self.go_to_Train)
        self.b4.place(x=width+400, y=380)

        self.logf.pack()
# Function called on Button clicked for Romur Page
    def go_to_Rumor(self):
            # will go to Romur Page
            reset_state(1)
# Function to go on train Page
    def go_to_Train(self):
        reset_state(2)



 #!!!!!!!!!!!!!! CAll At Romur PAge !!!!!!!!!!!!


class ThatPage():
    def __init__(self,master):
        print("in that page")
        # Window
        self.root = master
        self.root.title("Call at Romur")
       # self.root.option_add("*TCombobox*Background", 'black')

        # Some Usefull variables
        self.lang = StringVar()
        self.train_data = StringVar()
        self.type = IntVar()
        self.canvas_width = self.root.winfo_screenwidth()
        self.canvas_height = self.root.winfo_screenheight()
        self.python_green = "white"
        self.test_subject = StringVar()
        self.rumor=True
        # Create Widgets
        self.a1=None
        #Creating all widgets
        self.widgets()

    def widgets(self):

            self.logf = Canvas(self.root, bg='slateblue', height=self.canvas_height, width=self.canvas_width)


            points = [self.canvas_width, 450, self.canvas_width, self.canvas_height, 1000, self.canvas_height]
            self.logf.create_polygon(points, outline=self.python_green,
                                     fill='white', width=3)
            width=self.canvas_width/2-250
            Label(self.logf, text='CALL AT RUMOR ', fg="white", bg='slateblue', font=('arial 50 bold '), pady=5, padx=5) \
                .place(x=width, y=50)
            Label(self.logf, text='RUMOR? ', fg="white", bg='slateblue', font=('arial 30 bold '), pady=5, padx=5) \
                .place(x=width+170, y=150)

            Label(self.logf, text='LANGUAGE ', fg="white", bg='slateblue', font=('arial 18 bold '), pady=5, padx=5) \
                .place(x=self.canvas_width/2-400, y=330)

            language = ttk.Combobox(self.logf, width=27, textvariable=self.lang)

            # Adding combobox drop down list
            language['values'] = (' ENGLISH US(Default)',
                                  ' Hebrew',
                                  ' Russian',
                                  )
            language.grid(column=1, row=5)
            language.current(0)
            language.place(x=self.canvas_width/2-400, y=380)
            Label(self.logf, text='TRAIN DATA ', fg="white", bg='slateblue', font=('arial 18 bold '), pady=5, padx=5) \
                .place(x=self.canvas_width/2-100, y=330)
            training = ttk.Combobox(self.logf, width=27, textvariable=self.train_data)

            # Adding combobox drop down list
            training['values'] = (' English US V0.02(Newest)',
                                  ' Import Pre-Trained Data',
                                  )
            training.grid(column=1, row=5)
            training.current(0)
            training.place(x=self.canvas_width/2-100, y=380)
            Label(self.logf, text='TEST SUBJECT', fg="white", bg='slateblue', font=('arial  18 bold '), pady=5, padx=5) \
                .place(x=self.canvas_width/2+190, y=330)

            combostyle = ttk.Style()
            combostyle.configure('ARD.TCombobox', background="#ffcc66", fieldbackground="#ffff99")
            self.dataset = ttk.Combobox(self.logf, width=27, style='ARD.TCombobox', textvariable=self.test_subject)
            # Adding combobox drop down list
            self.dataset['values'] = (' Upload Text File',
                                 )
            self.dataset.grid(column=1, row=5)
            self.dataset.current(0)
            self.dataset.place(x=self.canvas_width/2+200, y=380)
            self.dataset.bind("<ButtonPress-1>",self.openFile)
# GO BUTTON INITIALIZATION HERE

            self.a1 = Button(self.logf, bg='slateblue', border=0,text='action')
            self.okb2 = PhotoImage(file="pics\go.png")
            self.tm2 = self.okb2.subsample(1, 1)
            self.a1.config(image=self.tm2)
            # Go Now Function is called on Button Click
            self.a1.bind("<ButtonPress-1>",self.go_now)
            self.a1_window=self.logf.create_window(self.canvas_width/2,500,window=self.a1)
# Romur or not Romur Images initialize
#             self.right = PhotoImage(file="pics/not_rumor.png")
#             self.wrong = PhotoImage(file="pics/rumor.png")
#
#
#             self.img=self.logf.create_image(self.canvas_width/2, 500, image=self.right)
            self.logf.pack()
            # self.logf.itemconfigure(self.img,state=tk.HIDDEN)



            #self.a1.place(x=570, y=480)
# Go back to main menu Button
            self.b3 = Button(self.logf, text="MENU>>", fg="white", font=('arial 20 bold'), bg='slateblue',
                             borderwidth=0,
                             command=self.go_back)
            self.b3.place(x=self.canvas_width-150, y=380)
            self.logf.pack()

#Go back to main menu function
    def go_back(self):
        reset_state(0)
# This is the function called when Go is clicked
    def go_now(self,event):

        #call predict function
        result, predict_index, img = main_predict(filename_predict)
        # show results by calling function
        self.show_Results(img,result)



    def show_Results(self,img,result):
        print(img)
        print("working")
        fg = 'white'
        self.right = PhotoImage(file=img)
        # self.wrong = PhotoImage(file="pics/rumor.png")

        self.img = self.logf.create_image(self.canvas_width / 2, 500, image=self.right)
        self.logf.pack()

        self.logf.delete(self.a1_window)
        self.logf.itemconfigure(self.img, state=tk.NORMAL)
        Label(self.logf, text=result, fg="white", bg='slateblue', font=('arial 12 bold '), pady=5, padx=5) \
            .place(x=self.canvas_width / 2.2, y=540)
 # this is the function which will get file
    def openFile(self,event):
            from tkinter.filedialog import askopenfilename
            global filename_predict
            print("file path here")
            # store ffile path in variable and use
            filename_predict = askopenfilename()
            print(filename_predict)



...
#!!!!!!!!!!!!!!! This is class to make circular progress bar !!!!!!!!!!!
# we Get an object as input and all these values defined from Train Page Below
class CircularProgressbar(object):
    def __init__(self,sc_width, canvas, x0, y0, x1, y1, width=2, start_ang=270, full_extent=360):
        # self.progressbar = CircularProgressbar(self.canvas_width,self.canvas, 500, 200, 1000, 650, 30)
        self.custom_font = tkFont.Font(family="Helvetica", size=20, weight='bold')
        self.style = ttk.Style()

        self.style.configure("text-color", foreground="red")
        self.canvas = canvas
        self.x0, self.y0, self.x1, self.y1 = x0+width, y0+width, x1-width, y1-width

        self.tx, self.ty = (x1-x0) / 2, (y1-y0) / 2
        self.sc_width=sc_width
        self.width = width
        self.start_ang, self.full_extent = start_ang, full_extent
        # draw static bar outline
        w2 = width / 2
        self.canvas.create_oval(self.x0-w2, self.y0-w2,
                                                self.x1+w2, self.y1+w2,fill="darkslateblue",outline='slateblue')

        self.canvas.create_oval(self.x0+w2, self.y0+w2,
                                                self.x1-w2, self.y1-w2,fill="slateBlue3",outline='slateblue')


        half_width=self.sc_width/2
#        print(self.half_width)
        new_width=self.x0 #change_here_if_need

        self.canvas.create_text(self.x0+x0/7.54, self.y0+y0/0.735, text='100',
                                                font=self.custom_font,fill="darkgray")
        self.canvas.create_text(new_width+x0/15.1, self.y0+y0/1.16, text='90',
                                font=self.custom_font,fill="darkgray")
        self.canvas.create_text(new_width+x0/7.54, self.y0+y0/2.45, text='75',
                                                font=self.custom_font,fill="darkgray")

        self.canvas.create_text(new_width+x0/3.177, self.y0+35, text='50',
                                                font=self.custom_font,fill="darkgray")
        self.canvas.create_text(new_width+x0/2.012, self.y0+y0/2.45, text='30',
                                                 font=self.custom_font,fill="darkgray")
        self.canvas.create_text(new_width+x0/1.77, self.y0+y0/1.16, text='20',
                                                 font=self.custom_font,fill="darkgray")
        self.canvas.create_text(new_width+x0/2.012, self.y0+y0/0.735, text='10',
                                                 font=self.custom_font,fill="darkgray")
       

        self.result=self.canvas.create_text(new_width+x0/3.177, self.y0+y0/0.63, text='0',
                                font=self.custom_font,fill="darkgray")
        print(x0,y0,x1,y1)
        self.running = False
# Starting the progress bar
    def start(self, interval=150):
        self.interval = interval
        self.increment = self.full_extent / interval
        self.extent = 0
        self.arc_id = self.canvas.create_arc(self.x0, self.y0, self.x1, self.y1,
                                             start=self.start_ang, extent=self.extent,
                                             width=self.width, style='arc',outline='cyan')
        self.percent = 0.0
      #  self.label_id = self.canvas.create_text(self.tx, self.ty, text=percent,
       #                                         font=self.custom_font)
        self.running = True
        self.canvas.after(interval, self.step, self.increment)
# Defining each step of progressbar here
    def step(self, delta):
        percent=float(self.percent)

        """Increment extent and update arc and label displaying how much complet:d."""
        if self.running and self.percent<100:
            print (self.percent)
            self.extent = (self.extent +delta) %360
            self.cur_extent = (self.extent + delta) % 360
            self.canvas.itemconfigure(self.arc_id, extent=self.cur_extent)
            self.percent=round(float(self.cur_extent) / self.full_extent * 100)
            text = '{:.0f}%'.format(self.percent)
            self.canvas.itemconfigure(self.result, text=str(text))

        self.after_id = self.canvas.after(self.interval, self.step, delta)

    def toggle_pause(self):
        self.running = not self.running
# !!!!!!!!!!This is the Train Page!!!!!!!!!!!!

class Train_Page:
    def __init__(self, master):
        self.root = master
        self.root.title("Call at Romur")


        self.canvas_width = self.root.winfo_screenwidth()
        self.canvas_height =self.root.winfo_screenheight()
        self.python_green = "white"
        self.running=False
        self.epoch=StringVar()
        self.loss=StringVar()
        self.accuracy=StringVar()
        self.epoch="0"
        self.loss="0.00"
        self.accuracy="00.00"


        self.createWidgets()
        self.root.mainloop()
# Creating all the widgets here
    def createWidgets(self):
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="slateBlue3")
        points = [self.canvas_width, 450, self.canvas_width, self.canvas_height, 1000, self.canvas_height]
        self.canvas.create_polygon(points, outline=self.python_green,
                                 fill='white', width=3)

        self.b1 = Button(self.canvas, text="<<MENU", fg="white", font=('arial 15 bold'), bg='slateBlue3',
                         borderwidth=0,
                         command=self.go_back)
        self.b1.place(x=50, y=self.canvas_height/2-50)

        self.b2 = Button(self.canvas, text="<ADVANCED", fg="white", font=('arial 15 bold'), bg='slateBlue3',
                         borderwidth=0,
                         command=self.advance_now)
        self.b2.place(x=self.canvas_width-150, y=self.canvas_height/2-50)

        self.a8 = Button(self.canvas, bg='RoyalBlue3', border=0)
        self.okb8 = PhotoImage(file="pics/loss.png")
        self.tm8 = self.okb8.subsample(1, 1)
        self.a8.config(image=self.tm8)

        width=self.canvas_width/2
        self.a8.place(x=width, y=50)
        self.loss_label = Label(self.canvas, text="LOSS", font=('times 20 bold italic'), bg='slateBlue3', fg='white')
        self.loss_label.place(x=width+40, y=40)
        self.loss_value = Label(self.canvas, text=self.loss, font=('times 20 bold italic'), bg='slateBlue3', fg='white')
        self.loss_value.place(x=width+60, y=70)

        width=self.canvas_width/2-200
        self.a81 = Button(self.canvas, bg='RoyalBlue3', border=0)
        self.okb81 = PhotoImage(file="pics/epoch.png")
        self.tm81 = self.okb81.subsample(1, 1)
        self.a81.config(image=self.tm81)
        self.a81.place(x=width, y=50)
        self.epoch_label = Label(self.canvas, text="EPOCH", font=('times 20 bold italic'),bg='slateBlue3', fg='white')
        self.epoch_label.place(x=width+40, y=40)
        self.epoch_value = Label(self.canvas, text=self.epoch, font=('times 20 bold italic'), bg='slateBlue3', fg='white')
        self.epoch_value.place(x=width+60, y=70)



        width=self.canvas_width/2+200
        self.a9 = Button(self.canvas, bg='RoyalBlue3', border=0)
        self.okb9 = PhotoImage(file="pics/accuracy.png")
        self.tm9 = self.okb9.subsample(1, 1)
        self.a9.config(image=self.tm9)
        self.a9.place(x=width, y=50)
        self.accuracy_label = Label(self.canvas, text="ACCURACY", font=('times 20 bold italic'), bg='slateBlue3', fg='white')
        self.accuracy_label.place(x=width+40, y=40)
        self.accuracy_value = Label(self.canvas, text=self.accuracy, font=('times 20 bold italic'), bg='slateBlue3', fg='white')
        self.accuracy_value.place(x=width+60, y=70)


        width=self.canvas_width/2-100
        self.a5 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb5 = PhotoImage(file="pics/play1.png")
        self.tm5 = self.okb5.subsample(1, 1)
        self.a5.config(image=self.tm5)
        self.a5_window=self.canvas.create_window(width,160,window=self.a5)

        #self.a5.place(x=width, y=120)


        width=width+70
        self.a6 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb6 = PhotoImage(file="pics/pause.png")
        self.tm6 = self.okb6.subsample(1, 1)
        self.a6.config(image=self.tm6)
        self.a6_window=self.canvas.create_window(width,160,window=self.a6)

        #self.a6.place(x=width, y=120)


        width = width + 70
        self.a7 = Button(self.canvas, bg='slateBlue3', border=0,command=self.pause)
        self.okb7 = PhotoImage(file="pics/stop.png")
        self.tm7 = self.okb7.subsample(1, 1)
        self.a7.config(image=self.tm7)
        self.a7_window=self.canvas.create_window(width,160,window=self.a7)
        #self.a7.place(x=width, y=120)


        width = width + 70
        self.a71 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb71 = PhotoImage(file="pics/reload.png")
        self.tm71 = self.okb71.subsample(1, 1)
        self.a71.config(image=self.tm71)
        self.a71_window=self.canvas.create_window(width,160,window=self.a71)



        # width=self.canvas_width/ 2.65
        from PIL import ImageTk , Image
        width = self.canvas_width
        height = self.canvas_height
        self.a1 = Button(self.canvas, bg='slateBlue3', border=0, command=self.start)
        self.img = Image.open("pics/begin2.PNG")
        self.img = self.img.resize((int(width*0.1075),int(height*0.1566)))
        self.okb1 = ImageTk.PhotoImage(self.img)
        self.a1.config(image=self.okb1)
        self.a1.place(x=0.4626*width ,y=height*.4055)



        width=self.canvas_width/2
        self.a2 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb2 = PhotoImage(file="pics/use_new.png")
        self.tm2 = self.okb2.subsample(1, 1)
        self.a2.config(image=self.tm2)
        self.a2_window=self.canvas.create_window(width-250,240,window=self.a2)

        #self.a2.place(x=width-250, y=200)

        self.a3 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb3 = PhotoImage(file="pics/export_new.png")
        self.tm3 = self.okb3.subsample(1, 1)
        self.a3.config(image=self.tm3)
        self.a3_window=self.canvas.create_window(width-310,350,window=self.a3)

       # self.a3.place(x=width-310, y=310)

        self.a4 = Button(self.canvas, bg='slateBlue3', border=0,command=self.openFile)
        self.okb4 = PhotoImage(file="pics/import.png")
        self.tm4 = self.okb4.subsample(1, 1)
        self.a4.config(image=self.tm4)
        self.a4.place(x=width+270, y=200)



        self.canvas.pack()
        # this will pas object to circularprogressbar class and will move there
        # self.progressbar = CircularProgressbar(self.canvas_width,self.canvas,self.canvas_width/2.1-200, 200, 1000, 650, 30) #change_here_if_need
        # screen_width = root.winfo_screenwidth()
        # screen_height = root.winfo_screenheight()
        # print(screen_width)
        # print(screen_height)
        # pixel_width = (12.18*screen_width) / 100
        # pixel_width2 = (15 * screen_width) / 100
        # pixel_height = (23.33 * screen_height) / 100
        # pixel_height2 = (27.5 * screen_height) / 100
        # print(pixel_height)


        # self.progressbar = CircularProgressbar(self.canvas_width, self.canvas, self.canvas_width / 2.65, self.canvas_height / 4.08,
        #                                    self.canvas_width /1.54, self.canvas_height / 1.35,self.canvas_width/53.33 )

        self.progressbar = CircularProgressbar(self.canvas_width, self.canvas, self.canvas_width *0.3768,
                                               self.canvas_height *0.2450,
                                               self.canvas_width * 0.6487, self.canvas_height *0.74,
                                               self.canvas_width / 53.33)
# When Begin Training is clicked

    def start(self):
        # making buttons enabled
        self.canvas.delete(self.a7_window)
        self.canvas.delete(self.a6_window)
        self.canvas.delete(self.a5_window)
        self.canvas.delete(self.a2_window)
        self.canvas.delete(self.a3_window)
        width = self.canvas_width / 2
        self.a2 = Button(self.canvas, bg='slateBlue3', border=0, command=self.openFile)
        self.okb2 = PhotoImage(file="pics/use1.png")
        self.tm2 = self.okb2.subsample(1, 1)
        self.a2.config(image=self.tm2)
        self.a2_window = self.canvas.create_window(width - 250, 240, window=self.a2)



        self.a3 = Button(self.canvas, bg='slateBlue3', border=0, command=self.openFile)
        self.okb3 = PhotoImage(file="pics/export.png")
        self.tm3 = self.okb3.subsample(1, 1)
        self.a3.config(image=self.tm3)
        self.a3_window = self.canvas.create_window(width - 310, 350, window=self.a3)

        width=self.canvas_width/2-100
        self.a5 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb5 = PhotoImage(file="pics/play.png")
        self.tm5 = self.okb5.subsample(1, 1)
        self.a5.config(image=self.tm5)
        self.a5_window=self.canvas.create_window(width,160,window=self.a5)




        width=width+70
        print(width)
        self.a6 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb6 = PhotoImage(file="pics/pause1.png")
        self.tm6 = self.okb6.subsample(1, 1)
        self.a6.config(image=self.tm6)
        self.a6_window=self.canvas.create_window(width,160,window=self.a6)



        width = width + 70

        self.a7 = Button(self.canvas, bg='slateBlue3', border=0)
        self.okb7 = PhotoImage(file="pics/stop1.png")
        self.tm7 = self.okb7.subsample(1, 1)
        self.a7.config(image=self.tm7)
        self.a7_window=self.canvas.create_window(width,160,window=self.a7)

        # Starting the progressbar
        self.progressbar.start()

        # Start training
        plt, time, start, no_of_epochs, acc, los = main_train()

        # This is the function to update epoch loss and accuracy values
        self.update_values(no_of_epochs, acc, los)
    # This is the function to update epoch loss and accuracy values
    def update_values(self,no_of_epochs,acc,los):
           # change each variable to update values here
           self.epoch= str(no_of_epochs)
           self.epoch_value['text']=self.epoch
           self.loss = los
           self.loss_value['text'] = self.loss
           self.accuracy= acc
           self.accuracy_value['text']=self.accuracy
    def pause(self):
        self.progressbar.toggle_pause()

    def go_back(self):
            reset_state(0)

    def openFile(self):
        from tkinter.filedialog import askopenfilename

        print("i am here")
        filename = askopenfilename()
        print(filename)
# Advance button clicked opens graph pictures
    def advance_now(self):
        newWindow = Toplevel(self.root)

        # sets the title of the
        # Toplevel widget
        newWindow.title("Advanced")
        ws = self.canvas_width
        hs = self.canvas_height
        # calculate position x, y
        x = (ws / 1) - (500 / 1)
        y = (hs / 1) - (700 / 1)
        newWindow.geometry('%dx%d+%d+%d' % (600, 700, x, y - 30))
        newWindow.attributes('-alpha', 0.7)
        canvas = Canvas(newWindow, width=600, height=700, bg="black")

        self.graph = PhotoImage(file="pics/graph_1.png")
        self.graph = self.graph.zoom(22)
        self.graph = self.graph.subsample(32)
        self.img = canvas.create_image(250, 200, image=self.graph)

        self.graph1 = PhotoImage(file="pics/graph_2.png")
        self.graph1 = self.graph1.zoom(22)
        self.graph1 = self.graph1.subsample(32)

        self.img1 = canvas.create_image(250, 520, image=self.graph1)

        canvas.pack()


# to switch between pages
def reset_state(x):
    global programState

    programState = x
    print(programState)
    refresh()
def refresh():
    print(programState)
    state=programState
    global current_page

    if current_page:
       print("in current page")
       for ele in root.winfo_children():
           ele.destroy()

    new_page_class = page_map[state]
    current_page = new_page_class(root)
#    current_page.pack(fill="both", expand=True)
if __name__ == "__main__":
    # 3 pages for app
    page_map = {0: ThisPage, 1: ThatPage, 2: Train_Page}
    current_page = 0
    root = Tk()
    reset_state(0)
    #ThisPage(root)
    root.mainloop()
    exit()