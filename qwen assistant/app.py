# Full AI Virtual Assistant Code with Dynamic QA Model and Additional Features
import speech_recognition as sr
import pyttsx3
import datetime
import pywhatkit
import webbrowser
import os
import requests
import psutil
from transformers import pipeline  # For Hugging Face models
import wikipedia  # For dynamic context
import threading
import time
import logging
import re
import google.auth
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress Wikipedia parser warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

# Configure logging
logging.basicConfig(filename="assistant.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# API Keys
WEATHER_API_KEY = "******"  # Replace with your WeatherAPI key
NEWS_API_KEY = "**********"  # Replace with your NewsAPI key
BING_SEARCH_API_KEY = "***********"  # Replace with your Bing Search API key

# TTS Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate

def speak(text):
    """Speak the given text."""
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def take_command():
    """Listen to user input and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"You: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't get that. Please repeat.")
        return ""
    except sr.RequestError:
        speak("Network error.")
        return ""

# Greeting Function
def greet_user():
    """Greet the user based on the time of day."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        speak("Good morning!")
    elif 12 <= hour < 18:
        speak("Good afternoon!")
    elif 18 <= hour < 22:
        speak("Good evening!")
    else:
        speak("Working late?")
    speak("I am your virtual assistant. How can I assist you today?")

# Hugging Face QA Model Function
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def ask_model(question):
    """Send a query to the Hugging Face QA model and get a response."""
    try:
        # Check if the question is too vague or incomplete
        if len(question.split()) < 3:  # Ensure the question has at least 3 words
            return "Please provide more details or complete your question."
        # Fetch dynamic context from Wikipedia
        search_results = wikipedia.search(question)
        if not search_results:
            return "Sorry, I couldn't find any information on that."
        # Use the most relevant result
        page_title = search_results[0]
        wikipedia_page = wikipedia.page(page_title)
        context = wikipedia_page.summary[:1000]  # First 1000 characters for more context
        # Generate response using the QA model
        response = qa_pipeline(question=question, context=context)
        answer = response["answer"].strip()
        # Add a fallback response
        if not answer or "sorry" in answer.lower():
            return "I'm not sure about that. Would you like me to search the web?"
        return answer
    except Exception as e:
        logging.error(f"Model Error: {e}")
        return "Sorry, I encountered an issue."

# Sentiment Analysis Function
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """Analyze the sentiment of the given text."""
    try:
        result = sentiment_pipeline(text)[0]
        return result['label'], result['score']
    except Exception as e:
        logging.error(f"Sentiment Analysis Error: {e}")
        return "Neutral", 0.5

# Weather Function
def get_weather(city):
    """Fetch real-time weather data using WeatherAPI."""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        condition = data["current"]["condition"]["text"]
        temp_c = data["current"]["temp_c"]
        feels_like = data["current"]["feelslike_c"]
        humidity = data["current"]["humidity"]
        return (
            f"The current weather in {city} is {condition}. "
            f"It is {temp_c}°C and feels like {feels_like}°C. "
            f"Humidity is {humidity}%."
        )
    except Exception as e:
        logging.error(f"Weather Error: {e}")
        return "Unable to fetch weather at the moment."

# Weather Forecast Function
def get_weather_forecast(city):
    """Fetch weather forecast data using WeatherAPI."""
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days=3"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        forecast_days = data["forecast"]["forecastday"]
        forecast_info = []
        for day in forecast_days:
            date = day["date"]
            condition = day["day"]["condition"]["text"]
            max_temp = day["day"]["maxtemp_c"]
            min_temp = day["day"]["mintemp_c"]
            avg_temp = day["day"]["avgtemp_c"]
            forecast_info.append(
                f"On {date}, expect {condition}. "
                f"High: {max_temp}°C, Low: {min_temp}°C, Average: {avg_temp}°C."
            )
        return "\n".join(forecast_info)
    except Exception as e:
        logging.error(f"Weather Forecast Error: {e}")
        return "Unable to fetch weather forecast at the moment."

# News Function
def get_news():
    """Fetch top news headlines using NewsAPI."""
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            return "Sorry, I couldn't fetch the news right now."
        articles = data.get("articles", [])[:5]
        if not articles:
            return "No news found at the moment. Please try again later."
        summaries = []
        for article in articles:
            title = article['title']
            description = article['description']
            summary = summarization_pipeline(description, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(f"{title}\n{summary}")
        return "Here are the top news headlines:\n" + "\n\n".join(summaries)
    except Exception as e:
        logging.error(f"News Error: {e}")
        return "Something went wrong while fetching the news."

# Web Search Function
def web_search(query):
    """Perform a web search and provide summarized results."""
    try:
        url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
        headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("webPages", {}).get("value", [])[:3]
        if not results:
            return "No results found for that query."
        summaries = []
        for result in results:
            title = result['name']
            snippet = result['snippet']
            summary = summarization_pipeline(snippet, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(f"{title}\n{summary}")
        return "Here are the top search results:\n" + "\n\n".join(summaries)
    except Exception as e:
        logging.error(f"Web Search Error: {e}")
        return "Something went wrong while performing the search."

# Battery Status Function
def get_battery_status():
    """Fetch battery percentage and charging status."""
    battery = psutil.sensors_battery()
    if battery is None:
        return "Battery information not available."
    percent = battery.percent
    plugged = battery.power_plugged
    status = "charging" if plugged else "not charging"
    return f"Battery is at {percent}% and is currently {status}."

# Reminder Function
def set_reminder(reminder_time_str, message):
    """Set a reminder using threading."""
    def reminder_thread():
        try:
            reminder_time = datetime.datetime.strptime(reminder_time_str, "%I:%M %p")
            now = datetime.datetime.now()
            reminder_time = reminder_time.replace(year=now.year, month=now.month, day=now.day)
            if reminder_time < now:
                reminder_time += datetime.timedelta(days=1)  # Set for the next day
            time_to_wait = (reminder_time - now).total_seconds()
            time.sleep(time_to_wait)
            speak(f"Reminder: {message}")
        except Exception as e:
            speak("Sorry, I couldn't set the reminder.")
            logging.error(f"Reminder Error: {e}")
    thread = threading.Thread(target=reminder_thread)
    thread.start()

# Calendar Integration
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

def get_calendar_service():
    """Authenticate and return a Google Calendar service object."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('calendar', 'v3', credentials=creds)
    return service

def add_event(title, date, time):
    """Add an event to the Google Calendar."""
    try:
        service = get_calendar_service()
        event = {
            'summary': title,
            'start': {
                'dateTime': f"{date}T{time}:00+05:30",
                'timeZone': 'Asia/Kolkata',
            },
            'end': {
                'dateTime': f"{date}T{time}:30+05:30",
                'timeZone': 'Asia/Kolkata',
            },
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        speak(f"Event {title} added to your calendar.")
    except Exception as e:
        logging.error(f"Calendar Error: {e}")
        speak("Sorry, I couldn't add the event.")

def get_upcoming_events():
    """Get upcoming events from the Google Calendar."""
    try:
        service = get_calendar_service()
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                              maxResults=10, singleEvents=True,
                                              orderBy='startTime').execute()
        events = events_result.get('items', [])
        if not events:
            return "No upcoming events found."
        event_list = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_list.append(f"{start} - {event['summary']}")
        return "Upcoming events:\n" + "\n".join(event_list)
    except Exception as e:
        logging.error(f"Calendar Error: {e}")
        return "Sorry, I couldn't fetch the events."

# Email Integration
EMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']

def get_email_service():
    """Authenticate and return a Gmail service object."""
    creds = None
    if os.path.exists('email_token.pickle'):
        with open('email_token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'email_credentials.json', EMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open('email_token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service

def send_email(to, subject, message_text):
    """Send an email using Gmail."""
    try:
        service = get_email_service()
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = 'me'
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body = {'raw': raw_message}
        message = service.users().messages().send(userId="me", body=body).execute()
        speak(f"Email sent to {to}.")
    except Exception as e:
        logging.error(f"Email Error: {e}")
        speak("Sorry, I couldn't send the email.")

def check_unread_emails():
    """Check for unread emails."""
    try:
        service = get_email_service()
        results = service.users().messages().list(userId='me', labelIds=['INBOX', 'UNREAD']).execute()
        messages = results.get('messages', [])
        if not messages:
            return "No unread emails."
        unread_count = len(messages)
        return f"You have {unread_count} unread emails."
    except Exception as e:
        logging.error(f"Email Error: {e}")
        return "Sorry, I couldn't check for unread emails."

# Music Control Functions
def control_music(action):
    """Control music playback (play, pause, skip)."""
    try:
        if action == "play":
            pywhatkit.playonyt("resume")
            speak("Resuming playback.")
        elif action == "pause":
            pywhatkit.playonyt("pause")
            speak("Pausing playback.")
        elif action == "skip":
            pywhatkit.playonyt("skip")
            speak("Skipping to the next track.")
        else:
            speak("I didn't understand that music control command.")
    except Exception as e:
        logging.error(f"Music Control Error: {e}")
        speak("Sorry, I couldn't control the music.")

# Main Assistant Logic
def run_assistant():
    """Run the virtual assistant."""
    greet_user()
    while True:
        command = take_command()
        if command:
            # Analyze sentiment of the command
            sentiment, score = analyze_sentiment(command)
            if sentiment == "POSITIVE":
                speak("Great! How can I assist you today?")
            elif sentiment == "NEGATIVE":
                speak("I'm here to help. What do you need assistance with?")
            # Process the command
            if 'time' in command:
                current_time = datetime.datetime.now().strftime('%I:%M %p')
                speak(f"The time is {current_time}")
            elif 'play' in command:
                song = command.replace('play', '').strip()
                if song:
                    speak(f"Playing {song} on YouTube.")
                    pywhatkit.playonyt(song)
                else:
                    speak("Please specify a song to play.")
            elif 'weather' in command:
                speak("Which city do you want the weather for?")
                city = take_command()
                if city:
                    weather_info = get_weather(city)
                    speak(weather_info)
                else:
                    speak("I didn't catch the city name.")
            elif 'weather forecast' in command:
                speak("Which city do you want the weather forecast for?")
                city = take_command()
                if city:
                    forecast_info = get_weather_forecast(city)
                    speak(forecast_info)
                else:
                    speak("I didn't catch the city name.")
            elif 'news' in command:
                speak("Fetching the latest news headlines.")
                news = get_news()
                speak(news)
            elif 'battery' in command or 'battery status' in command:
                status = get_battery_status()
                speak(status)
            elif 'set reminder' in command:
                speak("At what time should I remind you? Say the time in HH:MM AM or PM format.")
                reminder_time = take_command()
                speak("What should I remind you about?")
                message = take_command()
                if reminder_time and message:
                    set_reminder(reminder_time.upper(), message)
                    speak(f"Reminder set for {reminder_time} to {message}")
                else:
                    speak("Reminder time or message not understood.")
            elif 'send whatsapp' in command or 'whatsapp message' in command:
                speak("Whom do you want to message? Say the mobile number with country code.")
                number = take_command().replace(" ", "").replace("+", "")
                # Validate phone number
                pattern = r"^\+?[1-9]\d{1,14}$"  # E.164 format
                if re.match(pattern, number):
                    full_number = f"+{number}" if not number.startswith("+") else number
                    speak("What message should I send?")
                    message = take_command()
                    if message:
                        try:
                            pywhatkit.sendwhatmsg_instantly(full_number, message)
                            speak("Sending message now!")
                        except Exception as e:
                            logging.error(f"WhatsApp Error: {e}")
                            speak("Failed to send message.")
                    else:
                        speak("I didn't hear the message clearly.")
                else:
                    speak("That doesn't sound like a valid number.")
            elif 'open' in command:
                website = command.replace('open', '').strip()
                if website:
                    url = f"https://{website.lower().replace(' ', '')}.com"
                    speak(f"Opening {website}.")
                    webbrowser.open(url)
                else:
                    speak("Please specify a website to open.")
            elif 'search' in command:
                speak("What would you like to search for?")
                query = take_command()
                if query:
                    results = web_search(query)
                    speak(results)
                else:
                    speak("Please specify a search query.")
            elif 'add event' in command:
                speak("What is the title of the event?")
                title = take_command()
                speak("When is the event? Please say the date in YYYY-MM-DD format.")
                date = take_command()
                speak("At what time is the event? Please say the time in HH:MM format.")
                time = take_command()
                if title and date and time:
                    add_event(title, date, time)
                else:
                    speak("Event details not understood.")
            elif 'upcoming events' in command:
                events = get_upcoming_events()
                speak(events)
            elif 'send email' in command:
                speak("To whom should I send the email? Please say the recipient's email address.")
                to = take_command()
                speak("What is the subject of the email?")
                subject = take_command()
                speak("What is the message of the email?")
                message_text = take_command()
                if to and subject and message_text:
                    send_email(to, subject, message_text)
                else:
                    speak("Email details not understood.")
            elif 'check emails' in command:
                unread_emails = check_unread_emails()
                speak(unread_emails)
            elif 'play music' in command or 'pause music' in command or 'skip music' in command:
                if 'play' in command:
                    control_music("play")
                elif 'pause' in command:
                    control_music("pause")
                elif 'skip' in command:
                    control_music("skip")
            elif 'exit' in command or 'stop' in command:
                speak("Goodbye! Have a great day!")
                break
            else:
                speak("Let me check that for you.")
                try:
                    answer = ask_model(command)
                    speak(answer)
                except Exception as e:
                    speak("Sorry, I couldn't process that.")
        else:
            speak("Please say something.")

# Run the Assistant
if __name__ == "__main__":
    run_assistant()
