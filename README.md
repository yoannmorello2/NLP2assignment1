# NLP2assignment1
A learner that checks if captions correspond to images

## Part 1: collecting training data 
The initial idea was to use 10% of the 59000 images accompanied by a caption, that would be used as positive samples. Because of computations issues we reduced it progressively until 1000 captioned images (mainly because we first wanted to encode the caption with a BERT transformer, which was very expensive computationally).  After several experiments, we decided to keep only images which were related to a single category (most of the images in the databas are annotated in multi-categories). This was done because in view to determinate the bottleneck in our learner, we built to submodels based on our main model and trained them on classification tasks : one was to classify images by categories, one that was to classify captions.

To train our main classifier, we immediately produced 1000 false captions that would serve as negative samples. Generating random sequences of words could have led the NLP part of our model to learn to identify correctly formed sentences. Hence we prefered to load 1000 captions that are not among those used for the 1000 positive samples, and the captions corresponding to 1000 other images.


## Part 2: modeling 
Our model takes as input tupple of batches. The first element are batches of images  converted to be represented by 100x100x3 tensors of scaled values. The second element are batches of sequences of words padded to the length of the longest captions (25 in our last version) in our data and converted to 100 dim vectors using a gensim model prealably trained on all the captions available in the coco dataset. Its dimension is 25x100. The architecture is depicted below:


![ima-cap2 (1)](https://user-images.githubusercontent.com/98883383/197647495-572e1762-d971-47bd-b309-40476402497a.jpg)


## Part 3: training and testing 
The training function returns the loss function on every epoch. Every ten epochs, it calculates the validation loss and the training and validation accuracy. A test function calculate the accuracy of the model on any set of data. For computation efficiency, our batcher returns tupple of numpy arrays, which are only converted to tensor during the training loop.

## Part 4(and 5 bonus): evaluation and error analysis
As we can observe during the training, both the training and the validation loss decreases up to 40 epochs, but further the training loss keeps shrinking while the validation loss increases. This indicates overfitting. 3 simple strategies that we have not implemented could be used to tackle this problem : 1 decrease the size of the model ; 2 add drop out layers ; 3 train on more data.
Instead, we have chosen to analyse which part of the model is the bottleneck. So we decomposed it into 2 sub models(with the same hyperparameters and an architecture copied on our main model): one that we train to classify the images, the other to classify the captions, into one of the 90 categries names that figure in the annotation file.
The image classifier matches righly 54% of the images in the validation set after being trained 51 epochs (which seems pretty good as there are 90 categories). When training further the validation_acc decreases indicating overfitting as we our main model. Trained 51 epochs, the 1 layer LSTM scores also about 55% of accuracy on the validation set. We would have expected the caption classification task to be best handled. This could be expected as attributing a category to a caption should be a simple NLP task : context plays close to no role and even the order of the words in the sentence is not very important. The only thing a learner needs to 'learn' is to recognize words that are semanticaly close to each category name. 
To see if the NLP part of the task could be easily improved we tried a 2 layer LSTM model. This model needs to be trained longer, but it also reaches a best validation_acc in the 50% before overfitting.
From here we decided to leave the LSTMs without digging more in the hyperparameters (which we would have done if the purpose was to improve on our predictions), and we implemented an URN based on the article of Jean-Philippe Bernardy and Shalom Lappin : Unitary Recurrent Networks: Algebraic and Linear Structures for Syntax. As this architecture will be the core of our final project, we do not present it here but simply point out the results : In the simple version that we used, overfitting was nearly immediate in spite of the small size of the models used, and it lasted even when reducing this size. Our conclusion is that it memorizes the training data in ways that do not generalize.

