
# Text Steganography using Markov Chain

A text steganography method based on Markov chains is introduced, together with a reference implementation. This method allows for information hiding in texts that are automatically generated following a given Markov model. Other Markov - based systems of this kind rely on big simplifications of the language model to work, which produces less natural looking and more easily detectable texts. The method described here is designed to generate texts within a good approximation of the original language model provided.


## Working

[![flow.png](https://i.postimg.cc/Fz30GqPk/flow.png)](https://postimg.cc/dZQ7VWcJ)

The assignment works in three phases:

1)Pre-processing

2)Encoding

3)Decoding

Pre-processing phase:In this phase we use a predetermined text data.This source
text has to be predetermined between sender and receiver.This text can be taken
from Novels,Text Conversations,Newspapers,Textbooks etc.Once the data source
is finalized then we create the Unique markov chain for that predetermined text.

Encoding:
In the encoding process first we decide a n bit number which is going to be the size
of the longest input data.After this we create a continuous sequence of numbers
from 0 to n representing the subranges.Once these are finalised and generated we
will start moving from start state in markov chain based on the probability of the
next states.As we are moving down the chain we are also dividing out original
range into subranges of numbers as per the probability distribution and this process
continues till our subrange contains only a single number. Now the state path we
have followed from the start to the end state will represent the words used to
encode that n bit data. In this way we are going to encode all the input characters
one by one and their respective state path will give us the encoded data for those
characters

Decoding:
As from the encoding method we know for sure that the encoded messages are
formed due to a chain of words which are strung together using the makrov chain
upto a state where the subranges for the characters converge to a single value.Now
if we take the encoded message and start from the initial state and go further in the
chain using the set of words in the encoded text we reach a point where there is no
link for from the current state to the state for the next word.This is where we know
that the particular chain has ended at that particular state. Using this we can go to
the start of the chain and determine the character for which this chain was
developed.Similarly for every character a chain can be traced and the original
character can be determined from the encoded text message.Thus repeated
determinism of characters will lead to the generation of original characters and thus
the original plain text is obtained as a result.

The flowchart for Decoding is as follows:
[![flow2.png](https://i.postimg.cc/Zns08Lmv/flow2.png)](https://postimg.cc/CnDFYkzw)
## Try Out

The project is deployed online and can be accessed here:
[link]


## Screenshots
[![1.png](https://i.postimg.cc/pTqKSzxR/1.png)](https://postimg.cc/G8TB8BH5)
[![2.png](https://i.postimg.cc/FH8WWtpD/2.png)](https://postimg.cc/QFgJCyRT)
[![3.png](https://i.postimg.cc/9fPdGsn4/3.png)](https://postimg.cc/NyM2tCYY)


## Authors

- [Akshat Saxena](https://www.linkedin.com/in/akshat-saxena-6a3279188/)
Email: akshatsaxena977@gmail.com
