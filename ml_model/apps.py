from django.apps import AppConfig

class MlModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_model'

def getAllEmotions(prediction):
    return [ emotion_labels, prediction ]

def getTopTwoEmotions(prediction):
    emotion_list = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    top_index_1 = prediction.index(max(prediction)) # get index of highest value
    top_emotion_1 = emotion_list[top_index_1]
    emotion_1_data = prediction[top_index_1]
    prediction[top_index_1] = -1                    # reset highest value to -1
    top_index_2 = prediction.index(max(prediction)) # get new (2nd) highest value
    top_emotion_2 = emotion_list[top_index_2]
    emotion_2_data = prediction[top_index_2]
    prediction[top_index_1] = emotion_1_data
    return [top_emotion_1, top_emotion_2], [emotion_1_data, emotion_2_data]

def getPrediction(to_predict):
    # maybe add a try-catch for sentences longer than 66 words
    encoded_predict = encode_list([to_predict], word_lookup, max_sentence_len)
    if encoded_predict is None:
        return None
    Xnew = encoded_predict
    Ynew = neural_net.predict(Xnew, callbacks=[csv_logger]).tolist()[0]
    # Ynew takes the form of [[ %joy, %surprise, %sadness, %anger, %fear, %love' ]]
    return getAllEmotions(Ynew)

def emotionWheel(prediction):
    top_two_emotions, top_two_probs = getTopTwoEmotions(prediction)
    print("top two:",top_two_probs)
    # are top two adjacent on the wheel?
    # if so, are they in this order?
    # if not, reverse the order.
    # map it, then return the string.
    # Each array contains 10 items (index 0-9)
    # get a ratio as a decimal, multiply by 10 to get an int, 
    # and then just slap the int into the correct mapping and you're good

    # These results are already stored as decimals aka percentages...
    total = top_two_probs[0] + top_two_probs[1]
    ratio = round((top_two_probs[0]/total)*10)
    inverse_ratio = round((top_two_probs[1]/total)*10)

    print("Ratio:", ratio)
    print("Inverse Ratio:", inverse_ratio)

    if ratio < 0 or ratio > 9 or inverse_ratio < 0 or inverse_ratio > 9:
        print("aborted emotion wheel")
        return None # Prevents indexOutOfBounds, but the right math should prevent this
    
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

    if 'joy' in top_two_emotions and 'love' in top_two_emotions: 
        if top_two_emotions[0] == 'joy':
            return joy_to_love[ratio]
        else:
            return joy_to_love[inverse_ratio]
    elif 'love' in top_two_emotions and 'sadness' in top_two_emotions: 
        if top_two_emotions[0] == 'love':
            return love_to_sadness[ratio]
        else:
            return love_to_sadness[inverse_ratio]
    elif 'sadness' in top_two_emotions and 'anger' in top_two_emotions: 
        if top_two_emotions[0] == 'sadness':
            return sadness_to_anger[ratio]
        else:
            return sadness_to_anger[inverse_ratio]
    elif 'anger' in top_two_emotions and 'fear' in top_two_emotions: 
        if top_two_emotions[0] == 'anger':
            return anger_to_fear[ratio]
        else:
            return anger_to_fear[inverse_ratio]
    elif 'fear' in top_two_emotions and 'surprise' in top_two_emotions: 
        if top_two_emotions[0] == 'fear':
            return fear_to_surprise[ratio]
        else:
            return fear_to_surprise[inverse_ratio]
    elif 'surprise' in top_two_emotions and 'joy' in top_two_emotions: 
        if top_two_emotions[0] == 'surprise':
            return surprise_to_joy[ratio]
        else:
            return surprise_to_joy[inverse_ratio]
    else: return None

'''
test_list = [
    "hi my name is bob and i wanna kill george clooney",
    "beach balls really are just the worst",
    "if i could hug you it would be like a bomb going off in my heart",
    "sandwiches are mostly carbs",
    "why would anyone really write a book about vampires",
    "my penis is about the size you would suspect for a human"
    ]

for i in test_list:
    print(getPrediction(i))
'''

# TODO: Prevent the model from running twice every boot (research Django ready())
# https://medium.com/saarthi-ai/deploying-a-machine-learning-model-using-django-part-1-6c7de05c8d7
