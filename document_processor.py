import PyPDF2
import pandas as pd
from pathlib import Path
import json
import csv
from typing import Dict, List, Any

class DocumentProcessor:
    """Advanced document processing for multiple file types"""
    
    @staticmethod
    def process_pdf(file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                return {
                    "content": text.strip(),
                    "metadata": {
                        "source": file_path.name,
                        "type": "pdf",
                        "pages": len(reader.pages),
                        "size": len(text)
                    }
                }
        except Exception as e:
            return {"content": f"Error processing PDF: {e}", "metadata": {"source": file_path.name, "type": "pdf", "error": True}}
    
    @staticmethod
    def process_csv(file_path: Path) -> Dict[str, Any]:
        """Process CSV files into structured text"""
        try:
            df = pd.read_csv(file_path)
            
            # Convert to readable text format
            content = f"CSV Data from {file_path.name}:\n\n"
            content += f"Columns: {', '.join(df.columns)}\n"
            content += f"Rows: {len(df)}\n\n"
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                content += "Numeric Summary:\n"
                for col in numeric_cols:
                    content += f"- {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
                content += "\n"
            
            # Add sample rows
            content += "Sample Data:\n"
            content += df.head(10).to_string(index=False)
            
            return {
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "type": "csv",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "size": len(content)
                }
            }
        except Exception as e:
            return {"content": f"Error processing CSV: {e}", "metadata": {"source": file_path.name, "type": "csv", "error": True}}
    
    @staticmethod
    def process_excel(file_path: Path) -> Dict[str, Any]:
        """Process Excel files"""
        try:
            xl_file = pd.ExcelFile(file_path)
            content = f"Excel Data from {file_path.name}:\n\n"
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content += f"Sheet: {sheet_name}\n"
                content += f"Columns: {', '.join(df.columns)}\n"
                content += f"Rows: {len(df)}\n"
                content += df.head(5).to_string(index=False) + "\n\n"
            
            return {
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "type": "excel",
                    "sheets": xl_file.sheet_names,
                    "size": len(content)
                }
            }
        except Exception as e:
            return {"content": f"Error processing Excel: {e}", "metadata": {"source": file_path.name, "type": "excel", "error": True}}
    
    @staticmethod
    def process_markdown(file_path: Path) -> Dict[str, Any]:
        """Enhanced markdown processing with structure extraction"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract headers for metadata
            headers = []
            lines = content.split('\n')
            for line in lines:
                if line.startswith('#'):
                    headers.append(line.strip())
            
            return {
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "type": "markdown",
                    "headers": headers,
                    "lines": len(lines),
                    "size": len(content)
                }
            }
        except Exception as e:
            return {"content": f"Error processing Markdown: {e}", "metadata": {"source": file_path.name, "type": "markdown", "error": True}}
    
    @staticmethod
    def process_json(file_path: Path) -> Dict[str, Any]:
        """Process JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text
            content = f"JSON Data from {file_path.name}:\n\n"
            content += json.dumps(data, indent=2, ensure_ascii=False)
            
            return {
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "type": "json",
                    "keys": list(data.keys()) if isinstance(data, dict) else [],
                    "size": len(content)
                }
            }
        except Exception as e:
            return {"content": f"Error processing JSON: {e}", "metadata": {"source": file_path.name, "type": "json", "error": True}}
    
    @classmethod
    def process_document(cls, file_path: Path) -> Dict[str, Any]:
        """Process any supported document type"""
        suffix = file_path.suffix.lower()
        
        processors = {
            '.pdf': cls.process_pdf,
            '.csv': cls.process_csv,
            '.xlsx': cls.process_excel,
            '.xls': cls.process_excel,
            '.md': cls.process_markdown,
            '.markdown': cls.process_markdown,
            '.json': cls.process_json,
            '.txt': lambda p: {"content": p.read_text(encoding='utf-8'), "metadata": {"source": p.name, "type": "text", "size": p.stat().st_size}}
        }
        
        if suffix in processors:
            return processors[suffix](file_path)
        else:
            # Default text processing
            try:
                content = file_path.read_text(encoding='utf-8')
                return {
                    "content": content,
                    "metadata": {
                        "source": file_path.name,
                        "type": "unknown",
                        "size": len(content)
                    }
                }
            except:
                return {
                    "content": f"Unsupported file type: {suffix}",
                    "metadata": {"source": file_path.name, "type": "unsupported", "error": True}
                }