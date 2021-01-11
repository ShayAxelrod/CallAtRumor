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

matplotlib.use('Agg')  # Render *.PNG filetyped for high quality images using the Anti-Grain Geometry engine.

datasetDefaultAddress = "data/train.csv"
datasetAddress = datasetDefaultAddress

modelDefaultAddress = "modelCallRumor.h5"
modelAddress = modelDefaultAddress

# Load the training data file
def loadDataSet():
    import pandas as pd
    data = pd.read_csv(datasetAddress)  # Load the training data
    data = data.dropna()  # Drop null values
    data.reset_index(drop=True, inplace=True)  # Create new indexes for rows (drop=True) on current data (inplace=True)
    return data


# Cleans a text
def cleanText(text):  # This function clean text
    text = BeautifulSoup(text, "lxml").text  # Use as 'lxml' file
    text = text.replace('\n', '')  # Removes '\n'
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuations
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)  # Removes URLs
    text = text.lower()
    text = text.replace('x', '')  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAYBE REMOVE ME COMPLETELY!
    return text


# Applies TF-IDF on Batches
k = 100  # k most frequent words


def tfidt(batch):
    # k = 100  # k most frequent words
    countVectorizer = CountVectorizer(max_features=k)  # Construct a CountVectorizer with k features
    tfIdfTransformer = TfidfTransformer(use_idf=True)  # Construct a Tf-Idf Transformer

    wordCount = countVectorizer.fit_transform(batch)  # Count instances of the most popular k words
    newTfIdf = tfIdfTransformer.fit_transform(wordCount)  # Apply tf-idf on those k words

    trainWords = newTfIdf.toarray()  # Builds an array with the k most popular words ordered by TF-IDF
    return trainWords


# Applies attention
def applyAttention(wordVectorRowsInSentence):
    N_f = wordVectorRowsInSentence.shape[-1]  # N_f = k
    uiVectorRowsInSentence = tf.keras.layers.Dense(units=100, activation='tanh')(
        wordVectorRowsInSentence)  # [*, N_s, N_a] # Dense = Fully Connected
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
    return attentionWeightedSentenceVector  # output: [*, N_s, N_f]


def makeBatches(data):
    data_title = data['title'].to_numpy()  # Use the 'title' columns
    data_title = [cleanText(i) for i in data_title]  # Clean every title
    data_title = np.array(data_title)  # Turns dataSet into an array. Easier to work with arrays
    data_title = data_title[:18000]
    data_title = data_title.reshape(18, 1000)  # make 18 batch
    ###### SHAY Make this look less noob-like??
    # Apply tf-idf on every batch
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

    # Apply Attention on every IF-IDF matrix
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

    # Split Data:
    # Training: 15 Batches (83%)
    # Testing:  03 Batches (17%)
    x_train_2d = np.concatenate((X_train_tfidf_1, X_train_tfidf_2, X_train_tfidf_3, X_train_tfidf_4, X_train_tfidf_5,
                                 X_train_tfidf_6, X_train_tfidf_7, X_train_tfidf_8, X_train_tfidf_9, X_train_tfidf_10,
                                 X_train_tfidf_11, X_train_tfidf_12, X_train_tfidf_13, X_train_tfidf_14,
                                 X_train_tfidf_15), axis=0)
    x_test_2d = np.concatenate((X_train_tfidf_16, X_train_tfidf_17, X_train_tfidf_18), axis=0)

    # Converting the 2d arrays to 3d arrays. LSTM needs 3d arrays as input
    x_train = x_train_2d[:, None]
    x_test = x_test_2d[:, None]

    # data_label = labels of the dataSet. (1 = Rumor, 0 = NoRumor)
    data_label = data['label'].to_numpy()
    data_label = data_label[:18000]

    y_train = data_label[:15000]
    y_test = data_label[15000:]  ############# SHAY Limit to: 15000:18000 ??

    return x_train, x_test, y_train, y_test


# Build the RNN powered by LSTM model
def rnnWithLSTM():
    data_dim = k
    model = Sequential()
    model.add(LSTM(1024, input_shape=(None, data_dim), return_sequences=True))  # None = Dynamic Input Shape
    model.add(Dropout(0.3))  # Visible distances between training/validation. See output graphs.
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))  # Rumor/NonRumor
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def trainModel(x_train, x_test, y_train, y_test):
    # Halt the model if it does not improve for patience=5 epochs valued by 'val_loss'
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    # Save the best model valued by 'val_accuracy'
    global modelDefaultAddress
    checkpoint = ModelCheckpoint(modelDefaultAddress, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model = rnnWithLSTM()

    startTime = datetime.now()
    # Begin training the model.
    #  - history: Contains a record of training loss values and metrics values at successive epochs,
    #    as well as validation loss values and validation metrics values (if applicable).
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1000, callbacks=[early_stop, checkpoint])
    totalRunTime = datetime.now() - startTime

    return history, totalRunTime, startTime


# Displaying relevant data of a history object, including curves of loss and accuracy during training
def unpackHistoryAndBuildGraphs(history):
    import matplotlib.pyplot as plt

    # Unpack history
    history, accuracy, val_accuracy, loss, val_loss, epochs, numberOfEpochs = unpackHistory(history)
    drawHistoryGraphs(accuracy, val_accuracy, loss, val_loss, epochs)

    accuracy = val_accuracy[-1]  # Last Accuracy
    accuracy = "{:.2f}".format(accuracy)
    loss = val_loss[-1]  # Last Loss
    loss = "{:.2f}".format(loss)

    ######### SHAY Check if we need to return the library 'plt'. Can this be removed from the return?
    return plt, numberOfEpochs, accuracy, loss

def unpackHistory(history):
    history = history.history
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(accuracy) + 1)
    numberOfEpochs = len(accuracy)
    return history, accuracy, val_accuracy, loss, val_loss, epochs, numberOfEpochs

def drawHistoryGraphs(accuracy, val_accuracy, loss, val_loss, epochs):
    # Accuracy Graph
    plt.figure()
    plt.plot(epochs, accuracy, '-b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, '--ok', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('graphs/graphAccuracy.png')

    # Loss Graph
    plt.figure()
    plt.plot(epochs, loss, '-m', label='Training Loss')
    plt.plot(epochs, val_loss, '--ok', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # x = plt.show()  ###### SHAY THIS MAYBE USED TO DIPLAY IN REAL-TIME ???
    plt.savefig('graphs/graphLoss.png')

def main_train():
    dataSet = loadDataSet()
    x_train, x_test, y_train, y_test = makeBatches(dataSet)
    history, totalRunTime, startTime = trainModel(x_train, x_test, y_train, y_test)
    plt, no_of_epochs, acc, los = unpackHistoryAndBuildGraphs(history)
    return plt, totalRunTime, startTime, no_of_epochs, acc, los


def readFileAndAddTextToDataSet(fileAddress):
    # Load the data set
    global datasetAddress
    dataSet = pd.read_csv(datasetAddress)
    dataSet = dataSet.dropna()
    dataSet.reset_index(drop=True, inplace=True)

    # Read text from 'fileAddress'
    f = open(fileAddress, "r")
    textToPredict = f.read()

    # Generate DataFrame from the text and append it to the data set
    dataFrame = pd.DataFrame([[0, textToPredict, '', '', 0]], columns=['id', 'title', 'author', 'text', 'label'])
    newDataSet = dataSet.append(dataFrame, ignore_index=True)
    return newDataSet


def createPredictionBatchFromDataSet(dataSet):
    dataSetTexts = dataSet['title'].to_numpy()
    dataSetTexts = [cleanText(i) for i in dataSetTexts]  # Clean the data set
    dataSetTexts = np.array(dataSetTexts)
    batch = dataSetTexts[17500:]  # Create a batch from the last rows including our textToPredict row
    textToPredict_TFIDF = tfidt(batch)  # Apply TF-IDF
    textToPredict_2D = np.array(applyAttention(textToPredict_TFIDF))  # Apply Attention
    textToPredict = textToPredict_2D[:, None]  # Convert the 2D array to 3D array. LSTM needs a 3D array as input.
    return textToPredict


def predictTextFromBatch(batchToPredict):
    from tensorflow.keras.models import load_model
    global modelAddress
    model = load_model(modelAddress)
    y = -1  # Index of our text to predict within the batchToPredict (y=-1 == y=LastPosition)
    textToPredict = batchToPredict[y, None]
    # prediction = model.predict(batchToPredict[y, None])
    prediction = model.predict(textToPredict)
    predictionIndex = np.argmax(prediction[0])
    if predictionIndex == 0:
        predictionImage = 'pics/not_rumor.png'
        predictionResult = str(int(prediction[0][0] * 100)) + "% Non-Rumor\n" + str(
            int((1 - prediction[0][0]) * 100)) + "% Rumor"
    else:
        predictionImage = 'pics/rumor.png'
        predictionResult = str(int((1 - prediction[0][1]) * 100)) + "% Non-Rumor\n" + str(
            int(prediction[0][1] * 100)) + "% Rumor"
    return predictionResult, predictionIndex, predictionImage


def predictTextFromFile(fileAddress):
    dataSetWithTextToPredict = readFileAndAddTextToDataSet(fileAddress)
    batchToPredict = createPredictionBatchFromDataSet(dataSetWithTextToPredict)
    predictionResult, predictionIndex, predictionImage = predictTextFromBatch(batchToPredict)
    return predictionResult, predictionIndex, predictionImage


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


# SOME UI FUNCTIONS
def drawAxils(self):
    self.UI.create_oval(self.widthHalf - 5, self.heightHalf - 5, self.widthHalf + 5, self.heightHalf + 5,
                        fill='black')  # Dot in the middle
    self.UI.create_line(self.widthHalf, 0, self.widthHalf, self.canvasHeight, fill='black', width=5)  # Vertical Line
    self.UI.create_line(0, self.heightHalf, self.canvasWidth, self.heightHalf, fill='black', width=5)  # Horizonal Line


def createCanvas(self, master, title):
    self.root = master
    self.root.title(title)
    self.canvasWidth = self.root.winfo_screenwidth()
    self.canvasHeight = self.root.winfo_screenheight()
    self.widthHalf = self.canvasWidth / 2
    self.heightHalf = self.canvasHeight / 2
    self.python_green = "white"
    self.UI = Canvas(self.root, bg='#6A53E1', height=self.canvasHeight,
                     width=self.canvasWidth)  # Create a canvas window
    drawTriangle(self)  # Create a white triangle on the bottom right
    self.widgets()


def drawTriangle(self):
    # Create a white triangle on the bottom right
    #                X1                    Y1                      X2                 Y2                   X3                        Y3
    points = [self.canvasWidth, self.canvasHeight * 0.5, self.canvasWidth, self.canvasHeight, self.canvasWidth * 0.65,
              self.canvasHeight]
    self.UI.create_polygon(points, outline=self.python_green, fill='white', width=3)


def drawAlphaText(self):
    halfOfText = locateMiddleOfText(11, "Call At Rumor 2020 V0.01 alpha prototype")
    Label(self.UI, text="Call At Rumor 2020 V0.01 alpha prototype", fg="#CAD7EE", bg='#6A53E1',
          font=('Montserrat_Medium 11 italic')) \
        .place(x=self.widthHalf - halfOfText, y=self.canvasHeight - 120)


def locateMiddleOfText(fontSize, text):
    fontWidth = fontSize * 0.55
    halfTextLength = len(text) / 2
    halfOfText = halfTextLength * fontWidth
    return halfOfText


# ~~~~~~~~~~~~~~~ MAIN PAGE ~~~~~~~~~~~~~~~
class PageMain():
    def __init__(self, master):
        print("In Main Page")
        createCanvas(self, master, "Call At Rumor")

    def widgets(self):
        # Title
        halfOfText = locateMiddleOfText(60, "CALL AT RUMOR")
        Label(self.UI, text="CALL AT RUMOR", fg="white", bg='#6A53E1', font=('Antonio 60'), padx=5, pady=5) \
            .place(x=self.widthHalf - halfOfText, y=self.canvasHeight * 0.2)

        # Button Rumor
        offsetFromMiddle = 200
        halfOfText = locateMiddleOfText(28, "<< RUMOR?")
        self.buttonRumor = Button(self.UI, text="<< RUMOR?", fg="white", bg='#6A53E1', font=('Antonio 28'),
                                  borderwidth=0, command=self.gotoPageRumor)
        self.buttonRumor.place(x=self.widthHalf - halfOfText - offsetFromMiddle, y=self.canvasHeight * .35)

        # Button Train
        halfOfText = locateMiddleOfText(28, "TRAIN! >>")
        self.buttonTrain = Button(self.UI, text="TRAIN! >>", fg="white", bg='#6A53E1', font=('Antonio 28'),
                                  borderwidth=0, command=self.gotoPageTrain)
        self.buttonTrain.place(x=self.widthHalf - halfOfText + offsetFromMiddle, y=self.canvasHeight * .35)

        # Alpha
        drawAlphaText(self)

        self.UI.pack()

    def gotoPageRumor(self):
        reset_state(1)

    def gotoPageTrain(self):
        reset_state(2)


# ~~~~~~~~~~~~~~~ RUMOR PAGE ~~~~~~~~~~~~~~~
class PageRumor():
    def __init__(self, master):
        print("In Rumor Page")
        # self.root.option_add("*TCombobox*Background", 'black')
        self.lang = StringVar()
        self.train_data = StringVar()
        self.type = IntVar()
        self.test_subject = StringVar()
        self.rumor = True
        self.a1 = None
        createCanvas(self, master, "Call At Rumor")
        # drawAxils(self)

    def widgets(self):
        halfOfText = locateMiddleOfText(48, "CALL AT RUMOR")
        Label(self.UI, text="CALL AT RUMOR", fg="white", bg='#6A53E1', font=('Antonio 48'), padx=5, pady=5) \
            .place(x=self.widthHalf - halfOfText, y=self.canvasHeight * 0.05)

        halfOfText = locateMiddleOfText(28, "RUMOR?")
        Label(self.UI, text='RUMOR?', fg="white", bg='#6A53E1', font=('Antonio 28'), pady=5, padx=5) \
            .place(x=self.widthHalf - halfOfText - 10, y=self.canvasHeight * 0.13)

        halfOfText = locateMiddleOfText(18, "LANGUAGE")
        Label(self.UI, text='LANGUAGE', fg="white", bg='#6A53E1', font=('Montserrat_Medium 18'), pady=5, padx=5) \
            .place(x=self.widthHalf - halfOfText - 381, y=self.canvasHeight * 0.31)

        language = ttk.Combobox(self.UI, width=23, textvariable=self.lang, font=('Calibri 18'))
        language['values'] = (' English (US)',
                              ' Hebrew',
                              ' Russian')
        language.grid(column=1, row=5)
        language.current(0)
        language.place(x=self.widthHalf - 420, y=self.canvasHeight * 0.35)

        halfOfText = locateMiddleOfText(18, "TRAIN DATA")
        Label(self.UI, text='TRAIN DATA', fg="white", bg='#6A53E1', font=('Montserrat_Medium 18'), pady=5, padx=5) \
            .place(x=self.widthHalf - halfOfText - 58, y=self.canvasHeight * 0.31)

        training = ttk.Combobox(self.UI, width=23, textvariable=self.train_data, font=('Calibri 18'))
        training['values'] = (' English US V0.02 (latest)',
                              ' Import Pre-Trained Data')
        training.grid(column=1, row=5)
        training.current(0)
        training.place(x=self.widthHalf - 106, y=self.canvasHeight * 0.35)

        halfOfText = locateMiddleOfText(18, "TEST SUBJECT")
        Label(self.UI, text='TEST SUBJECT', fg="white", bg='#6A53E1', font=('Montserrat_Medium 18'), pady=5, padx=5) \
            .place(x=self.widthHalf - halfOfText + 262, y=self.canvasHeight * 0.31)

        combostyle = ttk.Style()
        combostyle.configure('TCombobox', background="#ffcc66", fieldbackground="#ffff99")
        self.dataset = ttk.Combobox(self.UI, width=23, textvariable=self.test_subject, font=('Calibri 18'),
                                    style='TCombobox')
        self.dataset['values'] = (' Upload Text File',)
        self.dataset.grid(column=1, row=5)
        self.dataset.current(0)
        self.dataset.place(x=self.widthHalf + 208, y=self.canvasHeight * 0.35)
        self.dataset.bind("<ButtonPress-1>", self.openFile)

        # GO BUTTON INITIALIZATION HERE
        self.okb2 = PhotoImage(file="pics\go1.png")
        self.tm2 = self.okb2.subsample(1, 1)
        self.a1 = Button(self.UI, bg='#6A53E1', border=0, text='action', image=self.tm2)

        # Event Handler
        self.a1.bind("<ButtonPress-1>", self.go_now)
        goButtonPlacement_Y = self.canvasHeight * 0.6
        self.a1_window = self.UI.create_window(self.canvasWidth / 2, goButtonPlacement_Y, window=self.a1)
        self.UI.pack()

        # Go Back To 'Menu' Button
        self.buttonMenu = Button(self.UI, text="MENU >>", fg="white", bg='#6A53E1', font=('Antonio 28'), borderwidth=0,
                                 command=self.go_back)
        self.buttonMenu.place(x=self.canvasWidth - 150, y=self.canvasHeight * .35)

        # Alpha
        drawAlphaText(self)

        self.UI.pack()

    # Go back to the menu page
    def go_back(self):
        reset_state(0)

    # GoButton eventHandler
    def go_now(self, event):
        # Call predict function
        predictionResult, predictionIndex, predictionImage = predictTextFromFile(filename_predict)
        # Display the results
        self.show_Results(predictionImage, predictionResult)

    def show_Results(self, predictionAddress, predictionResult):
        # print(img)
        print("Showing Results")
        self.right = PhotoImage(file=predictionAddress)

        resultCoordinate_Y = self.canvasHeight * 0.46
        self.img = self.UI.create_image(self.canvasWidth / 2, resultCoordinate_Y, image=self.right)
        self.UI.pack()

        self.UI.delete(self.a1_window)  # Remove Go Button
        self.UI.itemconfigure(self.img, state=tk.NORMAL)
        Label(self.UI, text=predictionResult, fg="white", bg='#6A53E1', font=('Montserrat_Medium 11'), pady=5, padx=5) \
            .place(x=(self.canvasWidth / 2.2) + 20, y=resultCoordinate_Y + 40)

    # Open File Dialog and return the address to the selected file
    def openFile(self, event):
        from tkinter.filedialog import askopenfilename
        global filename_predict
        filename_predict = askopenfilename()
        print(filename_predict)


class CircularProgressbar(object):
    def __init__(self, sc_width, canvas, x0, y0, x1, y1, width=2, start_ang=270, full_extent=360):
        self.canvas = canvas
        self.x0, self.y0, self.x1, self.y1 = x0 + width, y0 + width, x1 - width, y1 - width
        self.tx, self.ty = (x1 - x0) / 2, (y1 - y0) / 2
        self.sc_width = sc_width
        self.width = width
        self.start_ang, self.full_extent = start_ang, full_extent

        # Draw the 2 circles to form the darker inner circle
        w2 = width / 2
        self.canvas.create_oval(self.x0-w2, self.y0-w2, self.x1+w2, self.y1+w2, fill="cyan", outline='#6A53E1')
        self.canvas.create_oval(self.x0+w2, self.y0+w2, self.x1-w2, self.y1-w2, fill="#6A53E1", outline='#24304D')

        new_width = self.x0
        self.custom_font = tkFont.Font(family="Helvetica", size=20, weight='bold')
        self.canvas.create_text(self.x0+x0/7.54, self.y0+y0/0.735, text='12', font=self.custom_font, fill="white")
        self.canvas.create_text(new_width+x0/15.1, self.y0+y0/1.16, text='25', font=self.custom_font, fill="white")
        self.canvas.create_text(new_width+x0/7.54, self.y0+y0/2.45, text='37', font=self.custom_font, fill="white")
        self.canvas.create_text(new_width+x0/3.177, self.y0+45, text='50', font=self.custom_font, fill="white")
        self.canvas.create_text(new_width+x0/2.012, self.y0+y0/2.45, text='62', font=self.custom_font, fill="white")
        self.canvas.create_text(new_width+x0/1.77, self.y0+y0/1.16, text='75', font=self.custom_font, fill="white")
        self.canvas.create_text(new_width+x0/2.012, self.y0+y0/0.735, text='87', font=self.custom_font, fill="white")

        fontPercentage = tkFont.Font(family="Antonio", size=40)
        self.result = self.canvas.create_text(new_width+x0/3.177, self.y0+y0/0.63 - 20, text='0', font=fontPercentage, fill="white")

        self.running = False

    # Starting the progress bar
    # This is a reverse progressBar since arcs in python only grow counterclockwise.
    # So it is starting at 100% and going backwards to give us a clockwise feel
    def start(self, interval=150):
        self.interval = interval
        self.increment = self.full_extent / interval
        self.extent = 360
        self.arc_id = self.canvas.create_arc(self.x0, self.y0, self.x1, self.y1, start=self.start_ang, extent=self.extent, width=self.width, style='arc', outline='#24304D')
        self.percent = 0.0
        self.running = True
        self.canvas.after(interval, self.step, self.increment)

    # Defining each step of the progressbar
    def step(self, delta):
        if self.running and self.percent < 100:
            print(self.percent)
            self.extent = (self.extent - delta)
            self.canvas.itemconfigure(self.arc_id, extent=self.extent)

            self.percent = round(100-float(self.extent) / self.full_extent * 100)
            text = '{:.0f}%'.format(self.percent)
            self.canvas.itemconfigure(self.result, text=str(text))

        self.after_id = self.canvas.after(self.interval, self.step, delta)

    def toggle_pause(self):
        self.running = not self.running

    def stop(self):
        self.running = False
        self.extent = 360

# ~~~~~~~~~~~~~~~ TRAINING PAGE ~~~~~~~~~~~~~~~
class Train_Page:
    def __init__(self, master):
        self.root = master
        self.root.title("Call at Rumor")
        self.canvasWidth = self.root.winfo_screenwidth()
        self.canvasHeight = self.root.winfo_screenheight()
        self.widthHalf = self.canvasWidth / 2
        self.heightHalf = self.canvasHeight / 2
        self.python_green = "white"
        self.running = False
        self.epoch = StringVar()
        self.loss = StringVar()
        self.accuracy = StringVar()
        self.epoch = "00"
        self.loss = "0.00"
        self.accuracy = "0.00"
        self.createWidgets()
        self.root.mainloop()

    def createWidgets(self):

        self.UI = Canvas(self.root, bg='#6A53E1', height=self.canvasHeight, width=self.canvasWidth)  # Create a canvas window
        drawTriangle(self)  # Create a white triangle on the bottom right
        # Alpha
        drawAlphaText(self)

        points = [self.canvasWidth, 450, self.canvasWidth, self.canvasHeight, 1000, self.canvasHeight]

        self.b1 = Button(self.UI, text="<< MENU", fg="white", font=('Antonio 28'), bg='#6A53E1', borderwidth=0, command=self.go_back)
        self.b1.place(x=50, y=self.canvasHeight * .35)

        self.b2 = Button(self.UI, text="< ADVANCED", fg="white", font=('Antonio 28'), bg='#6A53E1', borderwidth=0, command=self.advance_now)
        self.b2.place(x=self.canvasWidth - 200, y=self.canvasHeight * .35)

        # TEXT FOR 'LOSS'
        self.okb8 = PhotoImage(file="pics/loss.png")
        self.tm8 = self.okb8.subsample(1, 1)
        self.a8 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm8)
        self.a8.place(x=self.widthHalf-40, y=50)

        textImageHeight = self.canvasHeight * 0.04
        self.loss_label = Label(self.UI, text="LOSS", font=('Montserrat_Medium 20'), bg='#6A53E1', fg='white')
        self.loss_label.place(x=self.widthHalf-10, y=textImageHeight+3)

        halfOfText = locateMiddleOfText(48, "0.00")
        self.loss_value = Label(self.UI, text=self.loss, font=('Antonio 48'), bg='#6A53E1', fg='white')
        self.loss_value.place(x=self.widthHalf - halfOfText + 20, y=78)

        # TEXT FOR 'EPOCH'
        self.okb81 = PhotoImage(file="pics/epoch.png")
        self.tm81 = self.okb81.subsample(1, 1)
        self.a81 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm81)
        self.a81.place(x=self.widthHalf - 240, y=50)

        self.epoch_label = Label(self.UI, text="EPOCH", font=('Montserrat_Medium 20'), bg='#6A53E1', fg='white')
        self.epoch_label.place(x=self.widthHalf - 210, y=textImageHeight+3)

        halfOfText = locateMiddleOfText(48, "00")
        self.epoch_value = Label(self.UI, text=self.epoch, font=('Antonio 48'), bg='#6A53E1', fg='white')
        self.epoch_value.place(x=self.widthHalf - halfOfText - 165, y=78)

        # TEXT FOR 'ACCURACY'
        self.okb9 = PhotoImage(file="pics/accuracy.png")
        self.tm9 = self.okb9.subsample(1, 1)
        self.a9 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm9)
        self.a9.place(x=self.widthHalf + 140, y=50)

        self.accuracy_label = Label(self.UI, text="ACCURACY", font=('Montserrat_Medium 20'), bg='#6A53E1', fg='white')
        self.accuracy_label.place(x=self.widthHalf + 170, y=textImageHeight + 3)

        halfOfText = locateMiddleOfText(48, "0.00")
        self.accuracy_value = Label(self.UI, text=self.accuracy, font=('Antonio 48'), bg='#6A53E1', fg='white')
        self.accuracy_value.place(x=self.widthHalf - halfOfText + 240, y=78)

        self.buttonHeight = self.canvasHeight*0.22
        # PLAY BUTTON
        self.okb5 = PhotoImage(file="pics/playClean.png")
        self.tm5 = self.okb5.subsample(1, 1)
        self.a5 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm5, command=self.start)
        self.a5_window = self.UI.create_window(self.widthHalf-121, self.buttonHeight, window=self.a5)

        # PAUSE BUTTON
        self.okb6 = PhotoImage(file="pics/pauseDirty.png")
        self.tm6 = self.okb6.subsample(1, 1)
        self.a6 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm6)
        self.a6_window = self.UI.create_window(self.widthHalf-38, self.buttonHeight, window=self.a6)

        # STOP BUTTON
        self.okb7 = PhotoImage(file="pics/stopDirty.png")
        self.tm7 = self.okb7.subsample(1, 1)
        self.a7 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm7, command=self.pause)
        self.a7_window = self.UI.create_window(self.widthHalf+40, self.buttonHeight, window=self.a7)

        # REPLAY BUTTON
        self.okb71 = PhotoImage(file="pics/replayDirty.png")
        self.tm71 = self.okb71.subsample(1, 1)
        self.a71 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm71)
        self.a71_window = self.UI.create_window(self.widthHalf+118, self.buttonHeight, window=self.a71)

        # BEGIN TRAINING BUTTON
        from PIL import ImageTk, Image
        self.img = Image.open("pics/beginTrainingClean.png")
        self.okb1 = ImageTk.PhotoImage(self.img)
        self.a1 = Button(self.UI, bg='#6A53E1', border=0, image=self.okb1, command=self.start)
        self.a1.place(x=self.widthHalf-110, y=self.heightHalf-117)

        # DBD: USE BUTTON
        #self.okb2 = PhotoImage(file="pics/useDirty.png")
        #self.tm2 = self.okb2.subsample(1, 1)
        #self.a2 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm2)
        #self.a2_window = self.UI.create_window(self.widthHalf - 250, self.canvasHeight*0.28, window=self.a2)

        # DBD: EXPORT BUTTON
        #self.okb3 = PhotoImage(file="pics/exportDirty.png")
        #self.tm3 = self.okb3.subsample(1, 1)
        #self.a3 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm3) # SHAY!!! REMOVE THE COMMAND FROM THERE. DEBUG ONLY
        #self.a3_window = self.UI.create_window(self.widthHalf - 341, self.canvasHeight*0.42, window=self.a3)

        # IMPORT BUTTON
        self.okb4 = PhotoImage(file="pics/importClean.png")
        self.tm4 = self.okb4.subsample(1, 1)
        self.a4 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm4, command=self.datasetChooser)
        self.a4.place(x=self.widthHalf + 203, y=self.canvasHeight*0.21)

        # FOR PROGRESSBAR
        x0 = self.canvasWidth * 0.365
        y0 = self.canvasHeight * 0.2450
        x1 = x0 + (self.canvasWidth * 0.2719)
        y1 = y0 + (self.canvasHeight * 0.495)
        self.progressbar = CircularProgressbar(self.canvasWidth, self.UI, x0, y0, x1, y1, self.canvasWidth / 53.33)

        self.UI.pack()

# When the "Begin Training" button is clicked
    def start(self):
        #DBD: self.UI.delete(self.a2_window)
        #DBD: self.UI.delete(self.a3_window)
        self.UI.delete(self.a5_window)
        self.UI.delete(self.a6_window)
        self.UI.delete(self.a7_window)

        # DBD: USE BUTTON
        #self.okb2 = PhotoImage(file="pics/useClean.png")
        #self.tm2 = self.okb2.subsample(1, 1)
        #self.a2 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm2, command=self.openFile)
        #self.a2_window = self.UI.create_window(self.widthHalf - 250, self.canvasHeight * 0.28, window=self.a2)

        # DBD: EXPORT BUTTON
        #self.okb3 = PhotoImage(file="pics/exportClean.png")
        #self.tm3 = self.okb3.subsample(1, 1)
        #self.a3 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm3)
        #self.a3_window = self.UI.create_window(self.widthHalf - 341, self.canvasHeight * 0.42, window=self.a3)

        # PLAY BUTTON
        self.okb5 = PhotoImage(file="pics/playDirty.png")
        self.tm5 = self.okb5.subsample(1, 1)
        self.a5 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm5)
        self.a5_window = self.UI.create_window(self.widthHalf - 121, self.buttonHeight, window=self.a5)

        # PAUSE BUTTON
        self.okb6 = PhotoImage(file="pics/pauseClean.png")
        self.tm6 = self.okb6.subsample(1, 1)
        self.a6 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm6, command=self.pause)
        self.a6_window = self.UI.create_window(self.widthHalf - 38, self.buttonHeight, window=self.a6)

        # STOP BUTTON
        self.okb7 = PhotoImage(file="pics/stopClean.png")
        self.tm7 = self.okb7.subsample(1, 1)
        self.a7 = Button(self.UI, bg='#6A53E1', border=0, image=self.tm7, command=self.stop)
        self.a7_window = self.UI.create_window(self.widthHalf + 40, self.buttonHeight, window=self.a7)

        # Starting the progressbar
        self.progressbar.start()

        # Start training
        plt, time, start, no_of_epochs, acc, los = main_train()

        # This is the function to update epoch loss and accuracy values
        self.update_values(no_of_epochs, acc, los)

    # This is the function to update epoch loss and accuracy values
    def update_values(self, no_of_epochs, acc, los):
        # change each variable to update values here
        self.epoch = str(no_of_epochs)
        self.epoch_value['text'] = self.epoch
        self.loss = los
        self.loss_value['text'] = self.loss
        self.accuracy = acc
        self.accuracy_value['text'] = self.accuracy

    def pause(self):
        self.progressbar.toggle_pause()

    def stop(self):
        self.progressbar.stop()
        reset_state(2)

    def go_back(self):
        reset_state(0)

    def datasetChooser(self):
        from tkinter.filedialog import askopenfilename
        global datasetAddress
        datasetAddress = askopenfilename()
        print(datasetAddress)

    def openFile(self):
        from tkinter.filedialog import askopenfilename
        filename = askopenfilename()
        print(filename)

    # Clicking on the advance button will open a small window with the graphs
    def advance_now(self):
        newWindow = Toplevel(self.root)
        newWindow.title("Advanced")

        wantedCanvasSize_x = 580
        wantedCanvasSize_y = 1020
        x = self.canvasWidth - wantedCanvasSize_x
        y = self.canvasHeight - wantedCanvasSize_y
        newWindow.geometry('%dx%d+%d+%d' % (wantedCanvasSize_x, wantedCanvasSize_y, x, y - 30))
        newWindow.attributes('-alpha', 0.7)
        canvas = Canvas(newWindow, width=wantedCanvasSize_x, height=wantedCanvasSize_y, bg="black")

        width_x = self.canvasWidth*0.15  # 250/1920
        height_y = self.canvasHeight*.205
        self.graph = PhotoImage(file="graphs/graphAccuracy.png")
        self.graph = self.graph.zoom(22)
        self.graph = self.graph.subsample(26)
        self.img = canvas.create_image(width_x, height_y, image=self.graph)

        self.graph1 = PhotoImage(file="graphs/graphLoss.png")
        self.graph1 = self.graph1.zoom(22)
        self.graph1 = self.graph1.subsample(26)
        self.img1 = canvas.create_image(width_x, height_y + 420, image=self.graph1)

        canvas.pack()

# to switch between pages
def reset_state(x):
    global programState
    programState = x
    print(programState)
    refresh()


def refresh():
    print(programState)
    state = programState
    global current_page
    if current_page:
        print("in current page")
        for ele in root.winfo_children():
            ele.destroy()

    new_page_class = page_map[state]
    current_page = new_page_class(root)

if __name__ == "__main__":
    # 3 pages for app
    page_map = {0: PageMain, 1: PageRumor, 2: Train_Page}
    current_page = 0
    root = Tk()
    reset_state(0)
    root.mainloop()
    exit()