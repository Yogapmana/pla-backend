from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

@tool
def youtube_transcript_tool(video_url: str) -> list[dict]:
    """Extract transcript from a YouTube video given its URL.
    Useful for getting educational content from video lectures or tutorials.
    Returns a single item list containing the transcript text and metadata."""
    try:
        # Extract video ID
        parsed_url = urlparse(video_url)
        if parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path[1:]
        elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                video_id = parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith('/embed/'):
                video_id = parsed_url.path.split('/')[2]
            elif parsed_url.path.startswith('/v/'):
                video_id = parsed_url.path.split('/')[2]
            else:
                raise ValueError("Invalid YouTube URL")
        else:
            raise ValueError("Invalid YouTube URL")

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['id', 'en'])
        
        # Combine transcript text
        full_text = " ".join([t.text for t in transcript.snippets])
        
        return [{
            "source_type": "youtube",
            "source_title": f"YouTube Video: {video_id}",
            "source_url": video_url,
            "raw_text": full_text,
            "relevance_score": 1.0 # default
        }]
    except Exception as e:
        print(f"YouTube transcript error for {video_url}: {e}")
        return []
