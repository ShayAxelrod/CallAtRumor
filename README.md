# CALL ATTENTION TO RUMORS: DEEP ATTENTION BASED RECURRENT NEURAL NETWORKS FOR EARLY RUMOR DETECTION
### In Partial Fulfillment of the Requirements for the Final Project in Software Engineering (Course 61771)
#### Authors:
- Shay Axelrod
- Yana Mamedov
#### Supervisor:
-  Prof. Zeev Volkovich

## Introduction
The rise in social media usage in recent years has made it a powerful platform for spreading rumors. The spread of rumors can pose a threat to cybersecurity, social, and economic stability worldwide. This project aims to detect the spread of rumors by identifying them on social networks in the early stages of their propagation in an automated way. RNNs have been proven to be effective in recent machine learning tasks for handling long sequential data. Three main challenges in early rumor detection must be addressed: (1) the system must adopt new features automatically and should not be hand-crafted; (2) the solution uses RNN. This algorithm has some well-known problems that prevent the processing of exceedingly long sequences of texts; (3) many duplications of posts with different contextual focuses must be handled. An attention-based RNN powered by long short-term memory (LSTM) with term frequency-inverse document frequency (tf-idf) mechanisms is proposed to overcome these challenges. In the project, the system detects rumors automatically using a deep attention model based on recurrent neural networks (RNN). For simplicity, the model is pre-trained using a dataset from social media sources. The model gets textual sequences of information from posts as input and constructs a series of feature matrices. Then, the RNN with an attention mechanism automatically learns new and hidden text representations. The attention mechanism is embedded in the system to help focus on specific words for capturing contextual variations of relevant posts over time. In the end, an additional hidden layer with a sigmoid activation function uses those text representations and predicts, as output, whether a text is a rumor or not. Furthermore, the system enables a trainer to train the algorithm with some new datasets.

#### For a better understanding of how this works, please read our [research paper](https://github.com/ShayAxelrod/CallAtRumor/blob/master/!!Capstone%20Project%20B-21-1-b-45/Capstone%20Project%20B-21-1-b-45.pdf).
#### To see our GUI in action, please watch our short video example: [![video](https://github.com/ShayAxelrod/CallAtRumor/blob/master/PrototypeFiles/Real%20GUI%20Images/Thumbnail.png)](https://youtu.be/h1GPNmdBXag)


## UML
#### Our Use Case Diagram ![Use Case Diagram](https://github.com/ShayAxelrod/CallAtRumor/blob/master/UML/UseCase.png)
#### Our Class Diagram ![Class Diagram](https://github.com/ShayAxelrod/CallAtRumor/blob/master/UML/classDiagramRNN.png)

#### For more information, test results, and more, read our [research paper](https://github.com/ShayAxelrod/CallAtRumor/blob/master/!!Capstone%20Project%20B-21-1-b-45/Capstone%20Project%20B-21-1-b-45.pdf).
