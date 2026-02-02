"""
Data ingestion module for journal entries.
Handles loading, validation, and preprocessing of journal data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class JournalEntry:
    """Represents a single journal entry."""
    
    def __init__(self, text: str, date: Optional[datetime] = None, metadata: Optional[Dict] = None):
        self.text = text
        self.date = date or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert entry to dictionary."""
        return {
            "text": self.text,
            "date": self.date.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "JournalEntry":
        """Create entry from dictionary."""
        return cls(
            text=data["text"],
            date=datetime.fromisoformat(data["date"]) if "date" in data else None,
            metadata=data.get("metadata", {})
        )


class JournalDataLoader:
    """Loads and manages journal entry data."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.entries: List[JournalEntry] = []
    
    def load_from_json(self, filename: str) -> List[JournalEntry]:
        """Load journal entries from JSON file."""
        filepath = self.data_path / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.entries = [JournalEntry.from_dict(entry) for entry in data]
            logger.info(f"Loaded {len(self.entries)} entries from {filename}")
            return self.entries
        
        except Exception as e:
            logger.error(f"Error loading file {filename}: {e}")
            return []
    
    def load_from_csv(self, filename: str, text_column: str = "text", date_column: str = "date") -> List[JournalEntry]:
        """Load journal entries from CSV file."""
        filepath = self.data_path / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        try:
            df = pd.read_csv(filepath)
            
            self.entries = []
            for _, row in df.iterrows():
                date = None
                if date_column in df.columns:
                    try:
                        date = pd.to_datetime(row[date_column])
                    except:
                        pass
                
                entry = JournalEntry(
                    text=row[text_column],
                    date=date,
                    metadata={k: v for k, v in row.items() if k not in [text_column, date_column]}
                )
                self.entries.append(entry)
            
            logger.info(f"Loaded {len(self.entries)} entries from {filename}")
            return self.entries
        
        except Exception as e:
            logger.error(f"Error loading file {filename}: {e}")
            return []
    
    def save_to_json(self, filename: str):
        """Save journal entries to JSON file."""
        filepath = self.data_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self.entries], f, indent=2)
            
            logger.info(f"Saved {len(self.entries)} entries to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
    
    def add_entry(self, entry: JournalEntry):
        """Add a new journal entry."""
        self.entries.append(entry)
    
    def get_entries_dataframe(self) -> pd.DataFrame:
        """Convert entries to pandas DataFrame."""
        if not self.entries:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "text": entry.text,
                "date": entry.date,
                **entry.metadata
            }
            for entry in self.entries
        ])
