import os
import json
from datetime import datetime
from typing import List, Dict, Any
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import subprocess
from pathlib import Path
from auto_file_sorter.auth_base import GoogleServiceAuth


class YouTubeWatchLaterReminder(GoogleServiceAuth):
    """Class to handle YouTube Watch Later playlist management"""

    @property
    def SCOPES(self) -> List[str]:
        return ['https://www.googleapis.com/auth/youtube.readonly']

    def __init__(self, credentials_path: str, token_path: str):
        """Initialize YouTube Watch Later reminder with OAuth2 credentials"""
        super().__init__(credentials_path, token_path)
        self.authenticate()

    def _build_service(self, credentials: Credentials) -> Any:
        """Build the YouTube API service"""
        return build('youtube', 'v3', credentials=credentials)

    def get_watch_later_playlist_id(self) -> str:
        """Get the Watch Later playlist ID."""
        try:
            next_page_token = None

            while True:
                # Set up request parameters
                params = {
                    'part': 'id,snippet',
                    'mine': False,
                    'maxResults': 50  # Maximum allowed per request
                }

                if next_page_token:
                    params['pageToken'] = next_page_token

                playlists = self.service.playlists().list(**params).execute()

                # Search for Watch Later playlist in current page
                for playlist in playlists.get('items', []):
                    if playlist['snippet']['title'] == 'Watch later':
                        return playlist['id']
                    else:
                        print(f"Playlist \"{
                              playlist['snippet']['title']}\" found, but it's not Watch Later")

                # Check if there are more pages
                next_page_token = playlists.get('nextPageToken')
                if not next_page_token:
                    break

            raise ValueError(
                "Watch Later playlist not found in any of your playlists")
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            raise

    def get_watch_later_videos(self, max_results: int = None, after_datetime: datetime = None) -> List[Dict[str, str]]:
        """Get videos from Watch Later playlist.

        Args:
            max_results (int, optional): Maximum number of videos to return. Defaults to None.
            after_datetime (datetime, optional): Only return videos added after this datetime. Defaults to None.

        Returns:
            List[Dict[str, str]]: List of video details sorted by date added (newest first)
        """
        try:
            playlist_id = self.get_watch_later_playlist_id()

            # Set up initial request parameters
            params = {
                'part': 'snippet',
                'playlistId': playlist_id,
                'order': 'date',  # Sort by date added
            }

            if max_results and not after_datetime:
                params['maxResults'] = max_results
            else:
                # If filtering by datetime, we need to fetch all items
                params['maxResults'] = 50  # Maximum allowed per request

            videos = []
            next_page_token = None

            while True:
                if next_page_token:
                    params['pageToken'] = next_page_token

                playlist_items = self.service.playlistItems().list(**params).execute()

                for item in playlist_items.get('items', []):
                    snippet = item['snippet']
                    video_added_at = datetime.strptime(
                        snippet['publishedAt'],
                        '%Y-%m-%dT%H:%M:%SZ'
                    )

                    # If we have a datetime filter and the video is older, we're done
                    if after_datetime and video_added_at <= after_datetime:
                        return videos

                    video = {
                        'title': snippet['title'],
                        'author': snippet['videoOwnerChannelTitle'],
                        'url': f"https://www.youtube.com/watch?v={snippet['resourceId']['videoId']}",
                        'added_at': video_added_at.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    videos.append(video)

                # If we have enough videos or there are no more pages, we're done
                if max_results and len(videos) >= max_results:
                    return videos[:max_results]

                next_page_token = playlist_items.get('nextPageToken')
                if not next_page_token:
                    break

            return videos

        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            raise

    def save_to_json(self, videos: List[Dict[str, str]]) -> None:
        """Save videos to JSON file."""
        data = {
            "youtube_watch_later_reminders": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "videos": videos
            }
        }

        json_file = self.script_dir / f"{__name__}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def create_apple_reminders(self, videos: List[Dict[str, str]]) -> None:
        """Create Apple Reminders for each video."""
        applescript = self.script_dir / f"{__name__}.applescript"

        script_content = '''
tell application "Reminders"
    set myList to list "YouTube Watch Later"
    
    -- Create list if it doesn't exist
    if myList is missing value then
        set myList to make new list with properties {name:"YouTube Watch Later"}
    end if
    
'''

        for video in videos:
            reminder_title = f'Youtube watch: "{
                video["title"]}" by author: "{video["author"]}"'
            script_content += f'''
    make new reminder at end of myList with properties {{name:"{reminder_title}", body:"{video["url"]}"}}
'''

        script_content += '''
end tell
'''

        # Save AppleScript
        with open(applescript, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Execute AppleScript
        subprocess.run(['osascript', str(applescript)], check=True)


def main(max_videos: int = 5):
    base_dir = Path(__file__).parent
    reminder = YouTubeWatchLaterReminder(
        credentials_path=str(base_dir / 'client_secrets.json'),
        token_path=str(base_dir / 'youtube_token.json')
    )
    videos = reminder.get_watch_later_videos(max_videos)
    reminder.save_to_json(videos)
    reminder.create_apple_reminders(videos)


if __name__ == '__main__':
    main()
