from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask import Flask, render_template, request, redirect
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO

from flask import Flask, render_template
import base64
import re
# NLP Packages
from textblob import TextBlob,Word 
import random 
import time
from flask import Flask, request, render_template
import PyPDF2, pdfplumber
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import io
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from flask_socketio import SocketIO, emit
import wave
import numpy as np
from scipy.io import wavfile
import os
from pydub import AudioSegment
import io
from io import BytesIO
from pydub.playback import play
import PyPDF2, pdfplumber
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# Connect to database
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
	return render_template('welcom2.html')

# Create tables for clients and companies if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    full_name TEXT NOT NULL
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    company_name TEXT NOT NULL
);
''')

conn.commit()

sia = SentimentIntensityAnalyzer()

@app.route('/cndidate_cap', methods=['GET', 'POST'])
def cndidate_cap():
    if request.method == 'POST':
        candidate_interview = request.form['candidate_interview']
        score = sia.polarity_scores(candidate_interview)['compound']

        if score >= 0.7:
            color = "green"
        elif score >= 0.4:
            color = "yellow"
        else:
            color = "red"

        return render_template("cap_score.html", score=score, color=color)
    return render_template("cndidate_cap.html")

@app.route("/company_login", methods=["GET", "POST"])
def company_login():
    if request.method == "POST":
        # Get form data
        username = request.form["username"]
        password = request.form["password"]

        # Check if username and password match with the database
        cursor.execute("SELECT * FROM companies WHERE username=? AND password=?", (username, password))
        company = cursor.fetchone()
        if company:
            # If username and password match, redirect to company dashboard
            return render_template("company_welcome.html")
        else:
            return "Invalid username or password"
    return render_template("company_login.html")

@app.route("/company_signup", methods=["GET", "POST"])
def company_signup():
    if request.method == "POST":
        # Get form data
        username = request.form["username"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        company_name = request.form["company_name"]

        # Check if password and confirm password match
        if password != confirm_password:
            return "Password and Confirm Password do not match"

        # Add company to database
        cursor.execute("INSERT INTO companies (username, password, company_name) VALUES (?, ?, ?)", (username, password, company_name))
        conn.commit()

        # If company is added successfully, redirect to company login page
        return redirect(url_for("company_login"))
    return render_template("company_signup.html")

@app.route("/client_login", methods=["GET", "POST"])
def client_login():
    if request.method == "POST":
        # Get form data
        username = request.form["username"]
        password = request.form["password"]

        # Check if username and password match with the database
        # Code to check username and password

        # If username and password match, redirect to client dashboard
        # return redirect(url_for("client_dashboard"))
        return render_template("client_welcome.html")
    return render_template("client_login.html")

@app.route("/client_signup", methods=["GET", "POST"])
def client_signup():
    if request.method == "POST":
        # Get form data
        username = request.form["username"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        full_name = request.form["full_name"]

        # Check if password and confirm password match
        if password != confirm_password:
            return "Password and Confirm Password do not match"

        # Code to add client to database

        # If client is added successfully, redirect to client login page
        # return redirect(url_for("client_login"))
        return "Client Signup Successful"
    return render_template("client_signup.html")

@app.route("/extractjob", methods=["GET", "POST"])
def jobdescription():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Read data from the uploaded file
            data = file.read().decode("utf8").lower()

            Haedings=[]
            # Extract headings from the data
            headings = re.findall(r'(?<=\n).*?(?=:)', data)
            Haedings.append(headings)

            # Split the data into sections based on the headings
            index=[]
            for i in headings:
                if i in data:
                    index.append(data.index(i))
            final_data=[]
            try:
                for i in range(len(index)):
                    final_data.append(data[index[i]:index[i+1]])
            except:
                final_data.append(data[index[-1]:])
            return render_template("extractjob.html", data=final_data)
    return render_template("client_welcome.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Read data from the uploaded file
            data = file.read().decode("utf8").lower()

            Haedings=[]
            # Extract headings from the data
            headings = re.findall(r'(?<=\n).*?(?=:)', data)
            Haedings.append(headings)

            # Split the data into sections based on the headings
            index=[]
            for i in headings:
                if i in data:
                    index.append(data.index(i))
            final_data=[]
            try:
                for i in range(len(index)):
                    final_data.append(data[index[i]:index[i+1]])
            except:
                final_data.append(data[index[-1]:])
            return render_template("extractjob.html", data=final_data)
    return render_template("client_welcome.html")

@app.route("/resumecomp", methods=['GET', 'POST'])
def resumecomp():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        if 'resume' in request.files:
            f = request.files['resume']

            # convert uploaded file to bytes-like object
            CV_File = f.read()
            with io.BytesIO(CV_File) as f:
                CV = PyPDF2.PdfReader(f)
                CV_pages = [CV.pages[i] for i in range(len(CV.pages))]

            Script = []
            with pdfplumber.open(io.BytesIO(CV_File)) as pdf:
                for i in range(0, len(CV_pages)):
                    page = pdf.pages[i]
                    text = page.extract_text(x_tolerance=2)
                    text_with_spaces = " ".join(text.split())
                    Script.append(text_with_spaces)

            Script=''.join(Script)
            CV_Clear=Script.replace("\n","")

            Script_Req=''.join(job_description)
            Req_Clear=Script_Req.replace("\n","")

            Match_Test=[CV_Clear,Req_Clear]

            cv=CountVectorizer()
            count_matrix=cv.fit_transform(Match_Test)

            MatchPercentage=cosine_similarity(count_matrix)[0][1]*100
            MatchPercentage=round(MatchPercentage,1)

            return render_template('resumecomp.html', match_percentage=MatchPercentage)
    return render_template('resumecomp.html', match_percentage=None)

#salary 
def calculate_salary(match_percent, min_salary, max_salary):
    salary_range = max_salary - min_salary
    salary = min_salary + (salary_range * (match_percent / 100))
    return salary

salary_ranges = [
    {"profession": "Doctor", "min_salary": 150000, "max_salary": 400000},
    {"profession": "Dentist", "min_salary": 100000, "max_salary": 250000},
    {"profession": "Pharmacist", "min_salary": 90000, "max_salary": 140000},
    {"profession": "Software developer", "min_salary": 60000, "max_salary": 140000},
    {"profession": "Registered nurse", "min_salary": 60000, "max_salary": 100000},
    {"profession": "Accountant", "min_salary": 50000, "max_salary": 120000},
    {"profession": "Teacher (K-12)", "min_salary": 40000, "max_salary": 90000},
    {"profession": "Mechanical engineer", "min_salary": 60000, "max_salary": 120000},
    {"profession": "Marketing manager", "min_salary": 90000, "max_salary": 200000},
    {"profession": "Machine learning engineer", "min_salary": 100000, "max_salary": 200000},
    {"profession": "ML Engineer", "min_salary": 100000, "max_salary": 200000},
    {"profession": "Artificial intelligence engineer", "min_salary": 110000, "max_salary": 250000},
    {"profession": "AI engineer", "min_salary": 110000, "max_salary": 250000}
]

def determine_salary(profession, match_percent):
    if not profession:
        return None

    for salary_range in salary_ranges:
        if salary_range["profession"] == profession:
            min_salary = salary_range["min_salary"]
            max_salary = salary_range["max_salary"]
            salary = calculate_salary(match_percent, min_salary, max_salary)
            salary = round(salary,1)
            return salary
        
    return None

def resume_matching(cv, job_description):
    # convert uploaded file to bytes-like object
    
    with io.BytesIO(cv) as f:
        CV = PyPDF2.PdfReader(f)
        CV_pages = [CV.pages[i] for i in range(len(CV.pages))]

    Script = []
    with pdfplumber.open(io.BytesIO(cv)) as pdf:
        for i in range(0, len(CV_pages)):
            page = pdf.pages[i]
            text = page.extract_text(x_tolerance=2)
            text_with_spaces = " ".join(text.split())
            Script.append(text_with_spaces)

    Script=''.join(Script)
    CV_Clear=Script.replace("\n","")

    Script_Req=''.join(job_description)
    Req_Clear=Script_Req.replace("\n","")

    Match_Test=[CV_Clear,Req_Clear]

    cv=CountVectorizer()
    count_matrix=cv.fit_transform(Match_Test)

    MatchPercentage=cosine_similarity(count_matrix)[0][1]*100
    MatchPercentage=round(MatchPercentage,1)

    return MatchPercentage

@app.route("/client_info", methods=['GET', 'POST'])
def client_info():
    return render_template('client_start.html')

def check_bias_and_discrimination(question):
    # Define a list of biased phrases
    bias_phrases = ["prefer candidates who are","What is your greatest weakness?",'hobby', 'personal', 'family', 'marital status', "looking for someone who is", "need someone who can","promoted beacuse","favorable treatment because",'will not be processed', 'only be', 'who fail', 'not allowed', 'should have', 'must have' , 'subject to', 'will not be considered', 'cannot be appointed', 'who lack']
    questions = [
          "What is your greatest weakness?",
          "What is something that you didn’t like about your last job?",
          "How do you deal w/ conflict with a go-worker?",
          "prefer candidates who are",
          "looking for someone who is",
          "promoted beacuse",
          "favorable treatment because",
          'will not be processed', 
          'only be', 
          'who fail', 
          'not allowed', 
          'should have', 
          'must have' , 
          'subject to', 
          'will not be considered', 
          'cannot be appointed', 
          'who lack',
          "need someone who can",
          "Why do you want this job?",
          "Why are you the best person for the job?",
          "Why are you leaving your job?",
          "Could you tell me about yourself and describe your background in brief?",
          "What type of work environment do you prefer?",
          "Your preferred environment should closely align to the company workplace culture? Do you prefer working independently or in a team?",
          "What are your salary expectations?",
          "Are you applying for other jobs?"
    ]
    
    # Define a list of discriminatory phrases
    discrimination_phrases = [" race ", " gender ", " age ", " sexual orientation ", " disability ", " national origin ", " religion "]

    # Iterate through the list of bias phrases
    for phrase in questions:
        # Use regular expressions to search for the phrase in the question
        match = re.search(phrase, question, re.IGNORECASE)
        
        # If a match is found, return "biased"
        if match:
            print(match)
            return 1
    
    # Iterate through the list of discrimination phrases
    for phrase in discrimination_phrases:
        # Use regular expressions to search for the phrase in the question
        match = re.search(phrase, question, re.IGNORECASE)
        
        # If a match is found, return "discriminatory"
        if match:
            print(match)
            return 1
    
    # If no matches are found for bias or discrimination, return "not biased or discriminatory"
    return 0

@app.route("/salaryrecomendation", methods=["POST"])
def salaryrecomendation():
    profession = request.form.get("profession")
    match_percent = int(request.form.get("match_percent", 0))
    salary = determine_salary(profession, match_percent)
    return render_template("salaryrecomendation.html", profession=profession, match_percent=match_percent, salary=salary)

@app.route("/interviewpage")
def interview():
    return render_template("interview.html")

@app.route("/clientchat")
def clientchat():
    return render_template("clientchat.html")

@app.route("/hrchat")
def hrchat():
    return render_template("hrchat.html")

# these will store the resume and the job description to hold on to them when page is reloaded
# during production, replace these with a database
resume_store = []
jd_store = []
secondary_jds = []

@socketio.on('load')
def handle_loading(json):
    emit('client_receive', {'data': json['data']}, broadcast=True)
    emit('hr_receive', {'data': json['data']}, broadcast=True)
    print(resume_store)
    if resume_store:
        emit('resume_sent', { 'text': resume_store[0]['text'].replace('\n', '<br>'), 'match_perc': resume_store[0]['matchperc'] }, broadcast=True)
    if jd_store:
        emit('jd_sent', { 'text': jd_store[0]['text'].replace('\n', '<br>'), 'jobtype': jd_store[0]['jobtype'] }, broadcast=True)

@socketio.on('resume_upload')
def handle_resume_upload(data):
    # Convert the array buffer to a bytes object
    match_percentage = resume_matching(io.BytesIO(data['data']).getvalue(), jd_store[0]['text'])
    pdf_bytes = io.BytesIO(data['data']).getvalue()

    # Use PyPDF2 library to extract the text from the PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)
    # match_percentage = resume_matching(io.BytesIO(data['data']).getvalue(), jd_store[0])
    print(match_percentage)
    resume_store.append({'text': text, 'matchperc': match_percentage})
    for i in secondary_jds:
        i['matchperc'] = resume_matching(io.BytesIO(data['data']).getvalue(), i['text'])

    # Emit the extracted text back to the client
    emit('resume_sent', { 'text': text.replace('\n', '<br>'), 'match_perc': match_percentage }, broadcast=True)

@socketio.on('jd_upload')
def handle_jd_upload(data):
    # Convert the array buffer to a bytes object
    pdf_bytes = io.BytesIO(data['data']).getvalue()

    # Use PyPDF2 library to extract the text from the PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)
    jd_store.append({'text': text, 'jobtype': data['jobtype']}) 
    # Emit the extracted text back to the client
    emit('jd_sent', { 'text': text.replace('\n', '<br>'), 'jobtype': data['jobtype'] }, broadcast=True)

# handle multiple secondary job descriptions
@socketio.on('secondary_jd')
def handle_jds(data):
    # Convert the array buffer to a bytes object
    pdf_bytes = io.BytesIO(data['data']).getvalue()

    # Use PyPDF2 library to extract the text from the PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)
    secondary_jds.append({'text': text, 'jobtype': data['jobtype'], 'matchperc' : 0})

@socketio.on('client_message')
def send_to_hr(json):
    print(json['data'])
    emit('hr_receive', {'data': json['data']}, broadcast=True)

@socketio.on('hr_message')
def send_to_hr(json):
    print(json['data'])
    # emit('client_receive', {'data': json['data']}, broadcast=True)
    hr_text = json['data']

    if check_bias_and_discrimination(hr_text):
        emit('hr_receive', {'data': "Your question is biased or discriminatory. This was not sent to the candidate."}, broadcast=True)
    else:
        language = 'en'
        hr_audio = gTTS(text = hr_text, lang=language, slow=False)
        hr_audio.save("hr_audio.wav")
        with open('./hr_audio.wav', 'rb') as ad:
            data = ad.read()
        emit('hr_audio', data, broadcast=True)

@socketio.on('audio')
def print_audio(audio_data):

    print("haha")
    print(len(audio_data))
    print(audio_data)
    audio_io = BytesIO(audio_data)
    audio_segment = AudioSegment.from_file(audio_io, format="webm", sample_width=2, frame_rate=48000, channels=2)
    audio_segment.export("audio.wav", format="wav")
    transcript = ""
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile('audio.wav')
    with audio_file as source:
        data = recognizer.record(source)
        transcript = recognizer.recognize_google(data, key=None)
        print(transcript)
        emit('hr_receive_transcript', {'data': transcript}, broadcast=True)

@socketio.on('interview_ended')
def interview_ended(data):
    print(data['data'])
    capability = sia.polarity_scores(data['data'])['compound']
    if capability >= 0.7:
        color = "green"
    elif capability >= 0.4:
        color = "yellow"
    else:
        color = "red"
    
    salary = determine_salary(data['job'], data['matchperc'])
    # for production, write capability to a database, then fetch capability from 
    # database in @app.route('/cndidate_cap')

    emit('client_receive', {'data': "Interview has ended! We'll get back to you soon."}, broadcast=True)
    emit('capability', {'capability': capability, 'color': color}, broadcast=True)
    emit('enable_backgroundcheck', {'capability': capability, 'salary': salary, 'matchperc': data['matchperc']}, broadcast=True)

# on background check
@socketio.on('bgcheckcomplete')
def on_completed_bgcheck(data):
    emit('completed', data, broadcast=True)

# on hired
@socketio.on('offersent')
def accept_or_no(data):
    emit('offerreceived', data, broadcast=True)

# on not hired
@socketio.on('nothired')
def other_jobs(data):
    emit('alternateoffers', { 'data': secondary_jds }, broadcast=True)

# on alternate offer
@socketio.on('alternateoffer')
def alt_offer(data):
    print(data)
    if data['yesOrno'] == 'no':
        print('no')
        emit('client_receive', { 'data': 'You did not receive an offer!' }, broadcast=True)
    else:
        emit('altoffer', data, broadcast=True)

@socketio.on('get_salary')
def getsalary(data):
    salary = determine_salary(data['job'], data['matchperc'])
    print(salary)
    emit('salary', {'data': salary}, broadcast=True)

# on questions uploaded
# extract questions from file
# convert to audio
# send to hr

@socketio.on('questions_upload')
def send_questions(data):
    #file1 = open("myfile.txt","r+")
 
    #print("Output of Read function is ")
    #print(file1.read())
    questions = str(data)

    if check_bias_and_discrimination(questions):
        emit('hr_receive', {'data': "Your question is biased or discriminatory. This was not sent to the candidate."}, broadcast=True)
    else:   
        language = 'en'
        
        questions = questions.replace('\\r', "")
        if (questions[0] == 'b' and questions[1] == "\'"):
            print('replace')
            questions = questions.replace("b'", "")

        questions = questions.split('\\n')
        
        i = 0
        for question in questions:
            i+=1
            hr_audio = gTTS(text = question, lang=language, slow=False)
            hr_audio.save("hr_audio.wav")
            with open('./hr_audio.wav', 'rb') as ad:
                data = ad.read()
            emit('hr_questions', {'data': question})
            emit('hr_audio', data, broadcast=True)
            print(question)

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0")
    #app.run()