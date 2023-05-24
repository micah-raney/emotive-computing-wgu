from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
import plotly.graph_objects as go
from .forms import *
from django.contrib.auth import authenticate
from django.contrib.auth import logout as django_logout
from django.contrib.auth import login as django_login
from django.contrib.auth.decorators import login_required
import logging
from .apps import generate_wordcloud, get_raw_data
from ml_model.neural_net_2 import predict, negative_emotions, all_negated_data, get_emotion_breakdown, get_sorted_wordlist
from ml_model.apps import emotionWheel, getTopTwoEmotions

title = "ThinkBetter Emotive Computing Project"
bgcolor='rgb(54,54,54)'
user_logger = logging.getLogger('user_logger')

def index(request):
    data_vis = {
        "wordcloud":"<a href='/data/wordcloud'>Word Cloud</a>", 
        "negation":"<a href='/data/negation'>Negation Analysis</a>", 
        "emotions":"<a href='/data/emotions'>Emotion Weighting</a>"
        }
    context = { 'pagetitle': title, 'data_vis':data_vis }
    if request.user.is_authenticated:
        context["authenticated_user"] = request.user.get_full_name()
    return render(request, 'webapp/index.html', context)

@login_required
def predict_view(request):
    form = PredictForm(request.POST)
    context = { 'predict_form':form }
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # check whether it's valid:
        if form.is_valid():
            x = form.get_sanitized_input()
            prediction = predict(x)
            if prediction is None:
                result = "Unfortunately, results were inconclusive. Please try another sentence."
                # TODO: log the input x
                result += " This issue has been logged."
                # the None is passed up from encode_list() if a word is not in the encoding dictionary. 
            else:
                top_two = getTopTwoEmotions(prediction)[0]
                result = top_two[0]+" and "+top_two[1]
                sub_emotion = emotionWheel(prediction)
                if sub_emotion is not None:
                    print(sub_emotion)
                    context['sub'] = sub_emotion
                emotion_list = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
                labels = emotion_list
                values = prediction
                print(labels, values)
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig.update_layout(paper_bgcolor=bgcolor)
                fig.update_layout(legend_font_color="white")
                piechart_html = fig.to_html(full_html=False)
                context['piechart'] = piechart_html
        else:
            result = "Input was invalid or empty. Please try again."
            # TODO: log the input x
        context['result'] = result
    return render(request, "webapp/predict.html", context)

def login(request):
    login_form = LoginForm()
    if request.method == 'POST':
        username = request.POST["user_field"]
        password = request.POST["pass_field"]
        pending_user = authenticate(request=request, username=username, password=password)
        if pending_user is not None:
            django_login(request, pending_user)
            user_logger.info(msg="Successfully logged in user \'%s\'" % pending_user.get_username())
            if not "next" in request.GET:
                return redirect(index)
            return redirect(request.GET["next"])
    context = { 'login_form': login_form }
    return render(request, 'webapp/login.html', context)

def logout(request):
    user_logger.info(msg="User \'%s\' logged out." % request.user.get_username())
    django_logout(request)
    # can't just call it logout because that's the view function...
    return redirect(index)

@login_required
def emotions(request):
    if request.user.username != "dev":
        return redirect(index)
    labels, values = get_emotion_breakdown()
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(paper_bgcolor=bgcolor)
    fig.update_layout(legend_font_color="white")
    piechart = fig.to_html(full_html=False)
    context = { 'piechart':piechart }
    return render(request, 'webapp/emotions.html', context)

@login_required
def wordcloud(request):
    topdown_list = get_sorted_wordlist("desc")
    most_common = generate_wordcloud(topdown_list)
    bottomup_list = get_sorted_wordlist("asc")
    least_common = generate_wordcloud(bottomup_list)
    context = { 'mostcommon':most_common, 'leastcommon':least_common }
    return render(request, 'webapp/wordcloud.html', context)

@login_required
def negation(request):
    if request.user.username != "dev":
        return redirect(index)
    base_case = []
    hypothesis = []
    for sentence, emotion in all_negated_data:
        if emotion in negative_emotions:
            hypothesis.append( (sentence, emotion) )
        else:
            base_case.append( (sentence, emotion) )
    
    fig = go.Figure(data=go.Bar(
        x=['Correlation', 'No Correlation'], y=[len(hypothesis), len(base_case)]))
    html = fig.to_html(config = {'displayModeBar': False}, full_html=False)
    context = { 'barchart':html }
    return render(request, 'webapp/negation.html', context)
