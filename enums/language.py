from enums.base_enum import BaseEnum

class Language(BaseEnum):
    # Latin family
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    POLISH = "pl"
    ROMANIAN = "ro"
    TURKISH = "tr"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"
    SWAHILI = "sw"
    FILIPINO = "tl"
    
    # CJK family
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja"
    KOREAN_CJK = "ko-cjk"  # Korean with CJK punctuation
    
    # Arabic family
    ARABIC = "ar"
    PERSIAN = "fa"
    URDU = "ur"
    PASHTO = "ps"
    
    # Devanagari family
    HINDI = "hi"
    MARATHI = "mr"
    SANSKRIT = "sa"
    NEPALI = "ne"
    
    # Cyrillic family
    RUSSIAN = "ru"
    UKRAINIAN = "uk"
    BULGARIAN = "bg"
    SERBIAN = "sr"
    KAZAKH = "kk"
    BELARUSIAN = "be"
    
    # Thai-Lao family
    THAI = "th"
    LAO = "lo"
    
    # Brahmic family
    BENGALI = "bn"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "kn"
    MALAYALAM = "ml"
    GUJARATI = "gu"
    PUNJABI = "pa"
    
    # Hebrew family
    HEBREW = "he"
    YIDDISH = "yi"
    
    # Hangul (modern Korean)
    KOREAN = "ko"
    
    # Greek
    GREEK = "el"
