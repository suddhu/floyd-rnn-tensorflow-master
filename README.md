# floyd-rnn-tensorflow-master
Tensorflow based RNN implementation that generates (awful) Pink Floyd lyrics for Python. I've put together the [Stink Floyd Fantasy LP](SF-fantasy-LP.md), where I've carefully curated my favorite outputs. 


It features the Multi-Layer Recurrent Neural Network, as used by Sherjil Ozair's [word-rnn-tensorflow](https://github.com/hunkim/word-rnn-tensorflow)<b><sup>1</sup></b>. While the training script remains the same, the sampling has been modified to best-represent lyric semantics. 

<b><sup>1</sup></b><i>This in turn is mostly derivative of Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).</i>

## Requirements 

- [Tensorflow 1.0.0](http://www.tensorflow.org)

## Training Data 

The lyrics can be found in <i>data/tinyshakespeare/input.txt</i> and was parsed from Vagalume.com.br via the [vagalume-download-lyrics](https://github.com/paladini/vagalume-download-lyrics) project. The entire [Pink Floyd discography](https://www.vagalume.com.br/pink-floyd/) comes to 200+ songs. The lyrics in <i>input.txt</i> have been manually pruned to remove unwanted symbols/characters. The sole intent of its use is for understanding the generation of lyrics with an RNN, no copyright infringement intended. 

## Usage
To train the network with default parameters on the floyd lyric datafile, run:
```bash
python train.py
```
Training may take some time, but upon completion you may sample from the trained model: 
```bash
python sample.py
```

## Sample output

Here's a few interested results, mildly formatted for the reader's pleasure. You can see much more at the [Stink Floyd Fantasy LP](SF-fantasy-LP.md). 
### A decent one
```
Slightly piper, stepping up - 
If I heard the rains. 
Will either hear the day trucks
Bottles on the lapel remember love.

Some little sand whisper
Packing shit now, saying don't.
Got so he never
Ice air - ooooh Maggie!
Love spoils the razor on,
Something my love at all it.

What away? A Fletcher. 
Most I've hold the sun -
With the pick the real
Going ons.

Another lifetime more do you shall?
Swan silent man, I
Show them an anything.
Eyes who was slip, of the
Tree surprise, petrified faces and needed a
Stone, don't love your time!
To my do to slied,
Looking of taken sound.
You're hello? Into the wall! With stormy
Hot machine gun, great!
S'il naked to will.
```

### A not-so-decent one
```
Can, combination, brighteyed
Ooh throng, throng Lee! Falling town,
Where we time out fits on freely.
Waking the long apprehension you're on
The river wondering bear. 
Used some gohills -
Waking without holds the loud
Pinky followed in all.

The colours with they
Rain, midnight gentlemen! Had wrong and
Hills off this want I hand.
With when to follow the
Doors even I've a
Hopeful I'm dont get.

Too surprised suns in formation, the
Damp in me for,
Is a way my
Faith to command at the east.

Boy, and let you feeling occasionally
On me, anybody up the time - anybody. 
In the film night,
Isn't in a dead day.
With history, in come with
Babe speak your church make waters!
```

You may tweak the parameters of <i>sample.py</i> and see how the lyrics shape up. The script also allocates words per line based on a probability distribution with assigned values. You are free to experiment with the RNN on the discography of other bands. 

## About

This project was created by Sudharshan Suresh, but it sources almost completely from the open-source projects mentioned previously. You are free to contact me with any issue/query/suggestion at suddhus at gmail dot com. 