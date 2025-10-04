from dataclasses import dataclass
from enums.language_family import LanguageFamily

@dataclass
class TextProcessingConfig:
    """Configuration for text processing based on language family."""
    character_range: str  # Regex for valid characters
    punctuation: str  # Regex for punctuation marks
    needs_lowercasing: bool  # Whether to lowercase
    has_word_spacing: bool  # Whether words are separated by spaces
    normalization_form: str  # Unicode normalization (NFKD, NFC, etc.)
    vocab_size: int # Recommended vocab size to have enough characters for tokens

def get_processing_config(family: LanguageFamily) -> TextProcessingConfig:
    """Get text processing configuration for a language family."""
    
    configs = {
        LanguageFamily.LATIN: TextProcessingConfig(
            character_range=r'a-z',
            punctuation=r'.?!,;:',
            needs_lowercasing=True,
            has_word_spacing=True,
            normalization_form='NFKD',
            vocab_size=10000
        ),
        
        LanguageFamily.CJK: TextProcessingConfig(
            character_range=r'\x{3040}-\x{309f}\x{30a0}-\x{30ff}\x{4e00}-\x{9faf}',
            punctuation=r'\x{3002}\x{ff1f}\x{ff01}\x{3001}',  # 。？！、
            needs_lowercasing=False,
            has_word_spacing=False,
            normalization_form='NFKD',
            vocab_size=30000
        ),
        
        LanguageFamily.ARABIC: TextProcessingConfig(
            character_range=r'\x{0600}-\x{06ff}',
            punctuation=r'\x{061f}\x{060c}\x{061b}',  # ؟،؛
            needs_lowercasing=False,
            has_word_spacing=True,
            normalization_form='NFKC',
            vocab_size=15000
        ),
        
        LanguageFamily.DEVANAGARI: TextProcessingConfig(
            character_range=r'\x{0900}-\x{097f}',
            punctuation=r'\x{0964}\x{0965}.?!,',  # ।॥ + Western
            needs_lowercasing=False,
            has_word_spacing=True,
            normalization_form='NFKD',
            vocab_size=15000
        ),
        
        LanguageFamily.CYRILLIC: TextProcessingConfig(
            character_range=r'\x{0400}-\x{04ff}',
            punctuation=r'.?!,;:\x{00ab}\x{00bb}',  # «»
            needs_lowercasing=True,
            has_word_spacing=True,
            normalization_form='NFKD',
            vocab_size=10000
        ),
        
        LanguageFamily.THAI_LAO: TextProcessingConfig(
            character_range=r'\x{0e00}-\x{0e7f}',
            punctuation=r'.?!,',
            needs_lowercasing=False,
            has_word_spacing=False,
            normalization_form='NFKC',
            vocab_size=15000
        ),
        
        LanguageFamily.BRAHMIC: TextProcessingConfig(
            character_range=r'\x{0980}-\x{09ff}\x{0b80}-\x{0bff}\x{0c00}-\x{0c7f}',
            punctuation=r'\x{0964}.?!,',  # । + Western
            needs_lowercasing=False,
            has_word_spacing=True,
            normalization_form='NFKD',
            vocab_size=15000
        ),
        
        LanguageFamily.HEBREW: TextProcessingConfig(
            character_range=r'\x{0590}-\x{05ff}',
            punctuation=r'.?!,;:',
            needs_lowercasing=False,
            has_word_spacing=True,
            normalization_form='NFKD',
            vocab_size=10000
        ),
        
        LanguageFamily.HANGUL: TextProcessingConfig(
            character_range=r'\x{ac00}-\x{d7af}',
            punctuation=r'.?!,',
            needs_lowercasing=False,
            has_word_spacing=True,
            normalization_form='NFKD',
            vocab_size=12000
        ),
        
        LanguageFamily.GREEK: TextProcessingConfig(
            character_range=r'\x{0370}-\x{03ff}',
            punctuation=r'.;!:',
            needs_lowercasing=True,
            has_word_spacing=True,
            normalization_form='NFKD',
            vocab_size=10000
        ),
    }
    
    return configs[family]
