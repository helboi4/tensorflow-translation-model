# language_mapper.py
from language import Language
from language_family import LanguageFamily

def get_language_family(language: Language) -> LanguageFamily:
    """Map a language to its punctuation/writing system family."""
    
    family_map = {
        # Latin
        Language.ENGLISH: LanguageFamily.LATIN,
        Language.SPANISH: LanguageFamily.LATIN,
        Language.FRENCH: LanguageFamily.LATIN,
        Language.GERMAN: LanguageFamily.LATIN,
        Language.ITALIAN: LanguageFamily.LATIN,
        Language.PORTUGUESE: LanguageFamily.LATIN,
        Language.DUTCH: LanguageFamily.LATIN,
        Language.POLISH: LanguageFamily.LATIN,
        Language.ROMANIAN: LanguageFamily.LATIN,
        Language.TURKISH: LanguageFamily.LATIN,
        Language.VIETNAMESE: LanguageFamily.LATIN,
        Language.INDONESIAN: LanguageFamily.LATIN,
        Language.MALAY: LanguageFamily.LATIN,
        Language.SWAHILI: LanguageFamily.LATIN,
        Language.FILIPINO: LanguageFamily.LATIN,
        
        # CJK
        Language.CHINESE_SIMPLIFIED: LanguageFamily.CJK,
        Language.CHINESE_TRADITIONAL: LanguageFamily.CJK,
        Language.JAPANESE: LanguageFamily.CJK,
        Language.KOREAN_CJK: LanguageFamily.CJK,
        
        # Arabic
        Language.ARABIC: LanguageFamily.ARABIC,
        Language.PERSIAN: LanguageFamily.ARABIC,
        Language.URDU: LanguageFamily.ARABIC,
        Language.PASHTO: LanguageFamily.ARABIC,
        
        # Devanagari
        Language.HINDI: LanguageFamily.DEVANAGARI,
        Language.MARATHI: LanguageFamily.DEVANAGARI,
        Language.SANSKRIT: LanguageFamily.DEVANAGARI,
        Language.NEPALI: LanguageFamily.DEVANAGARI,
        
        # Cyrillic
        Language.RUSSIAN: LanguageFamily.CYRILLIC,
        Language.UKRAINIAN: LanguageFamily.CYRILLIC,
        Language.BULGARIAN: LanguageFamily.CYRILLIC,
        Language.SERBIAN: LanguageFamily.CYRILLIC,
        Language.KAZAKH: LanguageFamily.CYRILLIC,
        Language.BELARUSIAN: LanguageFamily.CYRILLIC,
        
        # Thai-Lao
        Language.THAI: LanguageFamily.THAI_LAO,
        Language.LAO: LanguageFamily.THAI_LAO,
        
        # Brahmic
        Language.BENGALI: LanguageFamily.BRAHMIC,
        Language.TAMIL: LanguageFamily.BRAHMIC,
        Language.TELUGU: LanguageFamily.BRAHMIC,
        Language.KANNADA: LanguageFamily.BRAHMIC,
        Language.MALAYALAM: LanguageFamily.BRAHMIC,
        Language.GUJARATI: LanguageFamily.BRAHMIC,
        Language.PUNJABI: LanguageFamily.BRAHMIC,
        
        # Hebrew
        Language.HEBREW: LanguageFamily.HEBREW,
        Language.YIDDISH: LanguageFamily.HEBREW,
        
        # Hangul
        Language.KOREAN: LanguageFamily.HANGUL,
        
        # Greek
        Language.GREEK: LanguageFamily.GREEK,
    }
    
    return family_map[language]
