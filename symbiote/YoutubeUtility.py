#!/usr/bin/env python3
#
# YoutubeUtility.py

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
import webbrowser
import yt_dlp

class YouTubeUtility:
    def __init__(self, youtube_url):
        self.youtube_url = youtube_url
        self.video_id = self._extract_video_id()

    def _extract_video_id(self):
        """
        Extract the video ID from the YouTube URL.
        """
        try:
            url_data = urlparse(self.youtube_url)
            video_id = None

            if url_data.hostname == 'youtu.be':
                video_id = url_data.path[1:]
            elif url_data.hostname in ['www.youtube.com', 'youtube.com']:
                query = parse_qs(url_data.query)
                video_id = query.get('v')
                if video_id:
                    video_id = video_id[0]

            if not video_id:
                print("Error: Invalid YouTube URL. Video ID could not be extracted.")
                return None

            return video_id
        except Exception as e:
            print(f"Error extracting video ID: {str(e)}")
            return None

    def get_transcript(self, lang='en'):
        """
        Get captions/subtitles for the video in a specified language.
        """
        if not self.video_id:
            print("Error: No video ID available.")
            return None

        try:
            transcript = YouTubeTranscriptApi.get_transcript(self.video_id, languages=[lang])
            captions = ' '.join([entry['text'] for entry in transcript])
            return captions
        except TranscriptsDisabled:
            print("Subtitles/captions are disabled for this video.")
            return None
        except Exception as e:
            print(f"Error fetching transcript: {str(e)}")
            return None

    def play_video(self):
        """
        Open the YouTube video in the default web browser.
        """
        if not self.video_id:
            print("Error: No video ID available.")
            return None

        try:
            # Open the video URL in the default web browser
            webbrowser.open(self.youtube_url)
            return "Video opened in the default web browser."
        except Exception as e:
            print(f"Error playing video: {str(e)}")
            return None

    def get_video_info(self):
        """
        Fetch basic metadata for the video using yt-dlp.
        """
        if not self.video_id:
            print("Error: No video ID available.")
            return None

        try:
            with yt_dlp.YoutubeDL() as ydl:
                info_dict = ydl.extract_info(self.youtube_url, download=False)
                video_info = {
                    'title': info_dict.get('title', None),
                    'description': info_dict.get('description', None),
                    'channel_title': info_dict.get('uploader', None),
                    'publish_date': info_dict.get('upload_date', None),
                    'view_count': info_dict.get('view_count', None),
                    'length': info_dict.get('duration', None),
                    'thumbnails': info_dict.get('thumbnail', None)
                }
            return video_info
        except Exception as e:
            print(f"Error fetching video info: {str(e)}")
            return None

# Example usage:
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=o1dvcgyZqrc"
    utility = YouTubeUtility(youtube_url)

    # Fetch transcript
    transcript = utility.get_transcript()
    if transcript:
        print("Transcript:", transcript)

    # Play video
    result = utility.play_video()
    if result:
        print(result)

    # Get video info
    info = utility.get_video_info()
    if info:
        print("Video Info:", info)

