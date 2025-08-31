# emotive-computing-wgu

A backup of my capstone project with WGU - As of 2025 I have not bothered to ensure that it's in a functional state. 

The project uses a trained neural network combined with the Emotion Wheel Theory to attempt to intuit nuanced emotional sentiments from a given string of text. 

It is trained on a [dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) labeled with basic emotions - joy, love, fear, anger, sadness, and surprise. 


Then, when the user inputs text, a trained neural net makes a prediction as to which emotions best fit.
This returns a "best two" - let's say for example, surprise 65% joy 45%.
The ratio of the top two is saved - in this case, roughly 6:4. 

## The Emotion Wheel
The Emotion Wheel theory suggests that complex emotions can be modeled as a combination of more basic emotions. 
I took this idea and crafted my own simplistic Emotion Wheel, represented by the following spectra of nuanced emotions between two adjacent base emotions. 

From ml_model/apps.py: 
```
        joy_to_love = [
        'confident', 'optimistic', 'hopeful', 'respected', 'warm',
        'valued', 'peaceful', 'accepted', 'content', 'loving', 
        ]
    love_to_sadness = [ 
        'trusting', 'vulnerable', 'fragile', 'needy', 'powerless',
        'victimized', 'lonely', 'depressed', 'melancholy', 'mournful'
    ]
    sadness_to_anger = [
        'distraught', 'embarassed', 'worthless', 'disappointed', 'distant', 
        'withdrawn', 'dismissive', 'disrespected', 'frustrated', 'annoyed'
        ]
    anger_to_fear = [
        'irate', 'provoked', 'betrayed', 'indignant', 'violated', 
        'threatened', 'jealous', 'bitter', 'insecure', 'nervous'
        ]
    fear_to_surprise = [
        'anxious', 'overwhelmed', 'worried', 'helpless', 'stressed', 
        'rushed', 'perplexed', 'shocked', 'startled', 'confused'
        ]
    surprise_to_joy = [
        'excited', 'energetic', 'eager', 'amazed', 'free', 
        'playful', 'amused', 'interested', 'successful', 'proud'
        ]
```

The algorithm then takes its given ratio, say 6:4, and uses that to select a nuanced emotion from the list.

In this example, if there is slightly more surprise than joy, then the algorithm selects an emotion slightly closer to the beginning of the list (surprise) than the end (joy).
Perhaps 'free'. 

## Why the code doesn't work (and why there are no instructions on installation or usage)

NLP is pretty data-heavy, and some of the training files are considerably large. GitHub will not host sufficiently large files for just anybody. 
Hosting my own Git server would enable me to have Large File Storage (LFS) at no cost.
This project was also done a while ago, and I have not put in the upkeep to ensure that the various involved frameworks still work together. 
My submission of this project for university contained a pre-packaged release which is also too large to host here. 

I could fix these issues and offer a working release, but seeing as GPT was just starting to gain mass awareness at this time, it hardly seems like a worthwhile endeavor aside from the exercise in ML. 
It works... decently. But frankly there is just no substitute for practically unlimited access to data, computing time, and engineering. 
It was much more worth the coding practice and as a conversation piece on human-centric computing than as a functional product.

Nevertheless, if anyone expresses interest, I could get this thing working again, at least enough so for others to take up the torch and do something useful with the code. 
