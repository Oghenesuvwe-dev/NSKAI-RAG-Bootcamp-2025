import yt_dlp
import re
from langchain.schema import Document

def load_youtube_transcript(url):
    """Load and clean YouTube transcript using yt-dlp"""
    
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': True,
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Extract video info
            info = ydl.extract_info(url, download=False)
            
            # Get subtitles
            subtitles = info.get('subtitles', {})
            auto_subtitles = info.get('automatic_captions', {})
            
            # Try to get English subtitles
            transcript_text = ""
            
            if 'en' in subtitles:
                # Use manual subtitles if available
                for subtitle in subtitles['en']:
                    if subtitle.get('ext') in ['vtt', 'srv3', 'srv2', 'srv1']:
                        subtitle_url = subtitle['url']
                        transcript_text = _extract_text_from_subtitle_url(subtitle_url)
                        break
            elif 'en' in auto_subtitles:
                # Use auto-generated subtitles
                for subtitle in auto_subtitles['en']:
                    if subtitle.get('ext') in ['vtt', 'srv3', 'srv2', 'srv1']:
                        subtitle_url = subtitle['url']
                        transcript_text = _extract_text_from_subtitle_url(subtitle_url)
                        break
            else:
                raise Exception("No English subtitles found")
            
            # Clean the transcript
            cleaned_text = _clean_transcript(transcript_text)
            
            # Create document with metadata
            document = Document(
                page_content=cleaned_text,
                metadata={
                    'title': info.get('title', 'Unknown'),
                    'url': url,
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown')
                }
            )
            
            return [document]
            
        except Exception as e:
            raise Exception(f"Failed to load transcript: {str(e)}")

def _extract_text_from_subtitle_url(subtitle_url):
    """Extract text from subtitle URL"""
    import urllib.request
    import xml.etree.ElementTree as ET
    import json
    
    try:
        with urllib.request.urlopen(subtitle_url) as response:
            subtitle_data = response.read().decode('utf-8')
        
        # Try JSON format first (newer YouTube format)
        if subtitle_data.strip().startswith('{'):
            try:
                data = json.loads(subtitle_data)
                text_parts = []
                if 'events' in data:
                    for event in data['events']:
                        if 'segs' in event:
                            for seg in event['segs']:
                                if 'utf8' in seg:
                                    text_parts.append(seg['utf8'])
                return ' '.join(text_parts)
            except:
                pass
        
        # Try XML format
        try:
            root = ET.fromstring(subtitle_data)
            text_parts = []
            
            for text_elem in root.findall('.//text'):
                if text_elem.text:
                    text_parts.append(text_elem.text)
            
            return ' '.join(text_parts)
        except:
            pass
        
        # Fallback: treat as plain text
        return subtitle_data
    
    except Exception as e:
        raise Exception(f"Failed to extract subtitle text: {str(e)}")

def _clean_transcript(text):
    """Clean and normalize transcript text"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove timestamps and other artifacts
    text = re.sub(r'\d{1,2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # Clean up punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()