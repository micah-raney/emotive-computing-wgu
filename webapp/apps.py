from django.apps import AppConfig
from wordcloud import WordCloud

class WebappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'webapp'

def generate_wordcloud(dictionary):
    wc = WordCloud(width = 600, height = 500, max_words = 500).generate_from_frequencies(dictionary)
    return wc.to_svg()

def get_raw_data():
    '''gets plaintext string of all training data'''
    from ml_model.neural_net import load_raw, validation_file_path, test_file_path, train_file_path
    content = ""
    for path in [validation_file_path, test_file_path, train_file_path]:
        content += load_raw(path)
    return content

def add_user(username, password):
    from django.contrib.auth.models import User
    new_user = User.objects.create_user(username=username, password=password)
    
    # At this point, user is a User object that has already been saved
    # to the database. You can continue to change its attributes
    # if you want to change other fields.
    # example script below...
    #u = User.objects.get(username='test')
    #u.first_name="Test"
    #u.last_name="User"
    #u.save()
    #test_user = User.objects.create_user(username='dev', password='developer', first_name="Dev", last_name='Eloper')