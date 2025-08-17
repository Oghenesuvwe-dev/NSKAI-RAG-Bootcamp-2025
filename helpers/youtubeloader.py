from youtube_transcript_api import YouTubeTranscriptApi, Transcript
from langchain_core.documents import Document

def load_youtube_transcript(youtube_url: str) -> list[Document]:
    """Loads a YouTube video transcript and returns it as a list of Langchain documents."""
    
    # Validate URL
    if not youtube_url or "v=" not in youtube_url:
        raise ValueError("Invalid YouTube URL format. Must contain 'v=' parameter.")
    
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        
        if not video_id or len(video_id) != 11:
            raise ValueError(f"Invalid video ID extracted: {video_id}")
        
        # Use the list_transcripts method to get a list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find a human-generated transcript first, then an auto-generated one
        transcript: Transcript = transcript_list.find_transcript(['en'])
        
        # Fetch the actual transcript content
        transcript_data = transcript.fetch()
        
        if not transcript_data:
            raise ValueError("No transcript data found for this video")
        
        full_text = " ".join([d['text'] for d in transcript_data])
        
        if not full_text.strip():
            raise ValueError("Transcript is empty or contains no text")
        
        # Create a single Langchain Document
        doc = Document(page_content=full_text, metadata={"source": youtube_url})
        
        return [doc]
        
    except Exception as e:
        raise Exception(f"Failed to load YouTube transcript: {str(e)}")
