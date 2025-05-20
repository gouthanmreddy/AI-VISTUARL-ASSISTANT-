# AI Virtual Assistant

A comprehensive AI-powered virtual assistant with speech recognition, natural language processing, and multiple API integrations.

## Features

- Voice commands and text-to-speech responses
- Weather updates and forecasts
- News headlines with summarization
- Web search capabilities
- YouTube music playback
- WhatsApp messaging
- Email integration
- Calendar management
- Battery status monitoring
- Reminders and task scheduling
- Sentiment analysis
- Dynamic question answering using Hugging Face models

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API Keys:
   - WeatherAPI key
   - NewsAPI key
   - Bing Search API key

3. Set up Google Calendar and Gmail:
   - Place your `credentials.json` and `email_credentials.json` in the project root

## Usage

Run the assistant:
```bash
python app.py
```

Speak commands or questions naturally. The assistant will respond with voice and text output.

## Available Commands

- "What's the time?"
- "Play [song name]"
- "What's the weather in [city]?"
- "Get the weather forecast for [city]"
- "Show me the news"
- "Check battery status"
- "Set a reminder"
- "Send a WhatsApp message"
- "Open [website]"
- "Search for [query]"
- "Add calendar event"
- "Check upcoming events"
- "Send email"
- "Check emails"
- "Play/pause/skip music"

## Error Handling

The assistant includes comprehensive error handling and logging. Check `assistant.log` for detailed logs.

## Dependencies

See `requirements.txt` for a complete list of dependencies.