#!/usr/bin/env python3
#
# youtube_transcripts.py

import sys
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def get_youtube_transcript(youtube_url):
    # Extract video ID from the URL
    url_data = urlparse(youtube_url)
    video_id = None

    if url_data.hostname == 'youtu.be':
        video_id = url_data.path[1:]
    elif url_data.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(url_data.query)
        video_id = query.get('v')
        if video_id:
            video_id = video_id[0]

    if not video_id:
        return "Invalid YouTube URL"

    # Get the transcript using the video ID
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage:

youtube_url = "https://www.youtube.com/watch?v=wR9wYiEbu4I"
youtube_url = sys.argv[1]
transcript = get_youtube_transcript(youtube_url)
print(transcript)

