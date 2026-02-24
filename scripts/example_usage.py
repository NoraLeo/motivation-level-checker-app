"""
Example script demonstrating how to use the motivation level checker.
"""

from motivation_checker.models.mood_analyzer import MoodAnalyzer
from motivation_checker.data.preprocessing import TextPreprocessor
from motivation_checker.data.ingestion import JournalEntry, JournalDataLoader


def main():
    # Initialize components
    analyzer = MoodAnalyzer()
    preprocessor = TextPreprocessor()
    
    # Sample journal entries
    entries = [
        "Today was amazing! I finished all my tasks and feel super motivated to tackle tomorrow's challenges.",
        "Feeling really lazy today. Can't seem to get started on anything productive.",
        "Had a pretty normal day. Nothing special, just going through the motions.",
        "I'm so excited about my new project! Can't wait to dive in and make progress.",
        "Everything feels overwhelming. I don't have the energy to do anything."
    ]
    
    print("=" * 80)
    print("Motivation Level Checker - Example Analysis")
    print("=" * 80)
    print()
    
    for i, text in enumerate(entries, 1):
        print(f"Entry {i}:")
        print(f"Text: {text}")
        print()
        
        # Analyze the entry
        result = analyzer.analyze(text)
        
        print(f"Mood: {result['mood']}")
        print(f"Motivation Level: {result['motivation_level']:.2f}/100")
        print(f"Motivation Category: {result['motivation_category']}")
        print(f"Sentiment Polarity: {result['sentiment']['polarity']:.3f}")
        print(f"Sentiment Subjectivity: {result['sentiment']['subjectivity']:.3f}")
        print()
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()
