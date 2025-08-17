from youtube_transcript_api import YouTubeTranscriptApi, Transcript
from langchain_core.documents import Document

def load_youtube_transcript(youtube_url: str) -> list[Document]:
    """Loads a YouTube video transcript and returns it as a list of Langchain documents."""
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        
        # Use the list_transcripts method to get a list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find a human-generated transcript first, then an auto-generated one
        transcript: Transcript = transcript_list.find_transcript(['en'])
        
        # Fetch the actual transcript content
        transcript_data = transcript.fetch()
        
        full_text = " ".join([d['text'] for d in transcript_data])
        
        # Create a single Langchain Document
        doc = Document(page_content=full_text, metadata={"source": youtube_url})
        
        return [doc]
        
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return []
