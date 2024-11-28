from datetime import timedelta
import logging
import os
import re
import secrets
from typing import Literal
from dotenv import load_dotenv

from utils import str_to_enum

load_dotenv()

# echo: zsh $env | grep AI
# echo "zsh ./set_heroku_env.sh transcriby --remote=production"


class RegexConfig:
    REMINDER_FORMAT_START = re.compile(
        r'(?i)(?:set a reminder|remind me|i\'d like to be reminded)\s+', re.IGNORECASE)
    REMINDER_FORMAT = re.compile(
        r'(?i)(?:set a reminder|remind me|i\'d like to be reminded)\s+'
        r'(?:to\s+(?P<action>[^\[\]]+)\s+)?'
        r'(?:\[(?:Person|Contact):\s*(?P<contact_who>[^\]]+)\])?\s*'
        r'\[When:\s*(?P<when>[^\]]+)\]\s*'
        r'(?:about|to)?\s*'
        r'\[What:\s*(?P<about>[^\]]+)\]',
        re.IGNORECASE
    )
    EMAIL_IS_WHOLE_BODY_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    EMAIL_IS_IN_BODY_REGEX = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'


class EnvConfig:
    def __init__(self):
        # Prevent instantiation
        raise RuntimeError(
            "ReadOnlyConfig is a static class and should not be instantiated")
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    SAMPLE_VOICENOTE_BLOB = os.getenv(
        "SAMPLE_VOICENOTE_BLOB", "sample_voice_note.ogg")
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(24)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', '').replace('postgres://', 'postgresql+asyncpg://')
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TRANSCRIBY_CONTACT_CARD_URL = "https://firebasestorage.googleapis.com/v0/b/transcriby-b37f3.appspot.com/o/public%2FTranscriby.vcf?alt=media"
    TRANSCRIBY_CONTACT_CARD_URL_PATH = "public%2FTranscriby.vcf?alt=media"
    TWILIO_MESSAGE_LOG_URL = "https://console.twilio.com/us1/monitor/logs/sms/{account_sid}/{message_sid}?frameUrl=/console/sms/logs/{account_sid}/{message_sid}?x-target-region%3Dus1"

    ADMIN_API_KEY = os.getenv('ADMIN_API_KEY', "")
    LOGS_URL = os.getenv('LOGS_URL', "")
    USE_POSTGRES_DB = os.getenv('USE_POSTGRES_DB', "false").lower() == "true"
    _LOG_LEVEL_INITIAL = os.getenv('LOG_LEVEL_INITIAL', "INFO").upper()
    LOG_LEVEL_INITIAL = hasattr(logging, _LOG_LEVEL_INITIAL) and getattr(
        logging, _LOG_LEVEL_INITIAL) or logging.INFO
    # SECRET_KEY = os.environ.get('SECRET_KEY', "")
    # if not SECRET_KEY:
    #     raise ValueError("No SECRET_KEY")
    FREE_CREDITS_LIMIT = 10
    MIN_TOPUP_AMOUNT_PENCE = 100
    MAX_FREEMIUM_AUDIO_TRIM = 60
    MIN_FREEMIUM_AUDIO_TRIM = 30
    HARD_LIMIT_FREEMIUM = False
    """If true, then we hard limit freemium users to 10 free credits, after which they must subscribe; else we just apply a soft limit to discourage free users from uploading too many voice notes."""

    MAX_FREE_VOICE_NOTE_RECORDING_SECONDS = 600
    MAX_FREE_CREDITS = 10

    STATE_EXPIRATION_MINUTES = 10

    MAX_WA_MESSAGE_CHAR_LENGTH = 1550
    NEXT_TRANSCRIPTION_SEPARATOR = "\n\n---\n\n"

    SUPPORTED_LANGUAGES = ["English", "French"]

    # To calculate rates, we need to take our £2.50 / week for unlimited and set an amount of mins recording / amount of recordings count that we would to charge more then 2.50 for. This needs to be based on the number of voice notes per week. Say 10 / week.
    # at 3p / voice note less than 1min + 2p / min after that which combines transcription and analysis.
    # 10 * 3p = 30p
    # 10 * 2p = 20p
    # 30p + 20p = 50p / week
    # 50p / 10 = 5p / voice note
    # 5p / voice note * 10 = 50p / week
    # 50p / 10 = 5p / voice note
    # 5p / voice note * 10 = 50p / week
    # 50p / 10 = 5p / voice note
    # Total server costs per month are Heroku + OpenAI + Firebase + Google Cloud Storage + Twilio + Stripe + Domain + Email + WhatsApp Number etc...
    # We want to make 80% profit.
    # Heroku is $50 / month # TODO: Check email / dashboard for more accurate cost.
    # OpenAI is $50 / month # TODO: Check email / dashboard for more accurate cost.
    # Firebase is $100 / month # TODO: Check email / dashboard for more accurate cost.
    # Google Cloud Storage is $100 / month # TODO: Check email / dashboard for more accurate cost.
    # Twilio is $30 / month # TODO: Check email / dashboard for more accurate cost.
    # Stripe is $10 / month # TODO: Check email / dashboard for more accurate cost.
    # Domain is $10 / month # TODO: Check email / dashboard for more accurate cost.
    # Email is $10 / month # TODO: Check email / dashboard for more accurate cost.
    # WhatsApp Number is $5 / month # TODO: Check email / dashboard for more accurate cost.
    # This totals to $365 / month
    # 80% profit means we need to make $730 / month
    # If we have 30 subscription users, can we break even?
    # $730 / month / 30 = $24.33 / month per subscription user
    # $24.33 / month * 12 = $291.96 / year per subscription user
    # $291.96 / year / 52 = $5.62 / week per subscription user
    # $5.62 / week / 10 = $0.562 / week per voice note from subscription user
    # $0.562 / week * 4 = $2.248 / month per voice note from subscription user
    # $2.248 / month * 12 = $26.976 / year per voice note from subscription user
    # $26.976 / year / 52 = $0.518 / week per voice note from subscription user
    # $0.518 / week / 10 = $0.0518 / week per voice note from subscription user
    # $0.0518 / week * 4 = $0.2072 / month per voice note from subscription user
    # $0.2072 / month * 12 = $2.4864 / year per voice note
    # $730 / month / 10 = $73 / month per voice note
    # $73 / month * 12 = $876 / year per voice note
    # $216 / year / 52 = $4.15 / week per voice note
    # $4.15 / week / 10 = $0.415 / week per voice note
    # $0.415 / week * 4 = $1.66 / month per voice note
    # $1.66 / month * 12 = $19.92 / year per voice note
    # $19.92 / year / 52 = $0.383 / week per voice note
    _RATES_UNLIMITED_ADMIN = "25 pence per subscriber voicenote break even -> 25 pence per credit virtual value | Note we apply actual balance in PAYG model, so we just need to charge 25p per minute of transcription"
    _PAYG_PER_MINUTE_OF_TRANSCRIPTION_GBP = 0.25

    AUDIO_FILE_VOICENOTE_EXPECTED_FORMAT = "ogg"

    @staticmethod
    def get_PAYG_PER_MINUTE_OF_TRANSCRIPTION_GBP() -> float:
        return EnvConfig._PAYG_PER_MINUTE_OF_TRANSCRIPTION_GBP

    RATES_PAYG = \
        f"""£{_PAYG_PER_MINUTE_OF_TRANSCRIPTION_GBP}
            per transcribed voicenote minute with analysis bundled for free whilst I can."""
    RATES_UNLIMITED = \
        "£2.50 per week"

    TWILIO_ACCOUNT_SID = os.environ.get(
        'TWILIO_ACCOUNT_SID', "")  # or set directly
    TWILIO_AUTH_TOKEN = os.environ.get(
        'TWILIO_AUTH_TOKEN', "")    # or set directly

    OUR_TWILIO_NUMBER = os.environ.get('OUR_TWILIO_NUMBER', "")
    OUR_TWILIO_SANDBOX_NUMBER = os.environ.get('OUR_TWILIO_SANDBOX_NUMBER', "")
    ADMIN_PHONE = os.getenv("ADMIN_PHONE", "")
    STRIPE_PRIV_KEY_LIVE = os.getenv('STRIPE_PRIV_KEY_LIVE', "")
    STRIPE_PRIV_KEY_SANDBOX = os.getenv('STRIPE_PRIV_KEY_SANDBOX', "")
    USE_STRIPE_SANDBOX = os.getenv(
        'USE_STRIPE_SANDBOX', "false").lower() == "true"

    # Set OpenAI API key
    _OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    assert _OPENAI_API_KEY, "OPENAI_API_KEY is not set"
    # The above code is a Python comment. Comments in Python start with a hash symbol (#) and are used
    # to provide explanations or notes within the code. In this case, the comment "OPENAI" is followed
    # by a series of hash symbols (
    OPENAI_API_KEY = _OPENAI_API_KEY
    AI_MODEL: str = os.getenv('AI_MODEL', 'gpt-4o-mini')
    AI_IMAGE_MODEL: str = os.getenv('AI_IMAGE_MODEL', "dall-e-3")
    AI_TRANSCRIPTION_MODEL: str = os.getenv(
        'AI_TRANSCRIPTION_MODEL', 'whisper-1')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')

    SESSION_WINDOW_TIMEDELTA = timedelta(minutes=5)

    # Directory to save voice notes temporarily
    # ONLY FOR DEVELOPMENT PURPOSES

    # @classmethod
    # def VOICE_NOTES_DIR(cls) -> Literal['./voice_notes']:
    #     voice_notes_dir = './voice_notes'
    #     if not os.path.exists(voice_notes_dir):
    #         os.makedirs(voice_notes_dir)
    #     return voice_notes_dir

    GOOGLE_CLOUD_PUBLIC_DIR = '/public'
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")

    @staticmethod
    def get_GOOGLE_CLOUD_PUBLIC_DIR() -> str:
        return EnvConfig.GOOGLE_CLOUD_PUBLIC_DIR

    GIPHY_API_KEY = os.getenv('GIPHY_API_KEY', "")

    # Email regex pattern for validation

    REMINDER_FORMAT = RegexConfig.REMINDER_FORMAT
    EMAIL_IS_WHOLE_BODY_REGEX = RegexConfig.EMAIL_IS_WHOLE_BODY_REGEX
    EMAIL_IS_IN_BODY_REGEX = RegexConfig.EMAIL_IS_IN_BODY_REGEX
    REGEX_CONFIG = RegexConfig

    # Example pricing (check OpenAI for the latest numbers)
    # ~ https://openai.com/api/pricing/
    price_per_token = {
        "gpt-3.5-turbo": {"input": 2e-06, "output": 6e-06, "unit": "token"},
        "gpt-4": {"input": 3e-05, "output": 3e-05, "unit": "token"},
        "gpt-4-turbo": {"input": 1e-05, "output": 6e-05, "unit": "token"},
        "gpt-4o": {"input": 5e-06, "output": 1.5e-05, "unit": "token"},
        "gpt-4o-latest": {"input": 5e-06, "output": 1.5e-05, "unit": "token"},
        # $0.006 / minute (rounded to the nearest second)
        "whisper": {"second": 6e-09, "unit": "second_of_audio"}
    }

    @staticmethod
    def token_priced_models(): return [k for k in EnvConfig.price_per_token.keys(
    ) if "units" in EnvConfig.price_per_token[k] and EnvConfig.price_per_token[k]["units"] == "token"]

    # Logging EnvConfiguration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    LOG_TO_CONSOLE = os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true'
    LOG_TO_REMOTE = os.getenv('LOG_TO_REMOTE', 'false').lower() == 'true'
    REMOTE_LOGGING_URL = os.getenv('REMOTE_LOGGING_URL', '')


class ReadOnlyConfig(EnvConfig):
    """A readonly version of EnvConfig that provides property getters for all values"""

    def __init__(self):
        # Prevent instantiation
        # raise RuntimeError(
        #     "ReadOnlyConfig is a static class and should not be instantiated")
        pass

    @property
    def SECRET_KEY(self) -> str:
        return EnvConfig.SECRET_KEY

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return EnvConfig.SQLALCHEMY_DATABASE_URI

    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        return EnvConfig.SQLALCHEMY_DATABASE_URL

    @property
    def SQLALCHEMY_TRACK_MODIFICATIONS(self) -> bool:
        return EnvConfig.SQLALCHEMY_TRACK_MODIFICATIONS

    @property
    def TRANSCRIBY_CONTACT_CARD_URL(self) -> str:
        return EnvConfig.TRANSCRIBY_CONTACT_CARD_URL

    @property
    def TRANSCRIBY_CONTACT_CARD_URL_PATH(self) -> str:
        return EnvConfig.TRANSCRIBY_CONTACT_CARD_URL_PATH

    @property
    def TWILIO_MESSAGE_LOG_URL(self) -> str:
        return EnvConfig.TWILIO_MESSAGE_LOG_URL

    @property
    def ADMIN_API_KEY(self) -> str:
        return EnvConfig.ADMIN_API_KEY

    @property
    def FREE_CREDITS_LIMIT(self) -> int:
        return EnvConfig.FREE_CREDITS_LIMIT

    @property
    def MIN_TOPUP_AMOUNT_PENCE(self) -> int:
        return EnvConfig.MIN_TOPUP_AMOUNT_PENCE

    @property
    def MAX_FREEMIUM_AUDIO_TRIM(self) -> int:
        return EnvConfig.MAX_FREEMIUM_AUDIO_TRIM

    @property
    def MIN_FREEMIUM_AUDIO_TRIM(self) -> int:
        return EnvConfig.MIN_FREEMIUM_AUDIO_TRIM

    @property
    def HARD_LIMIT_FREEMIUM(self) -> bool:
        return EnvConfig.HARD_LIMIT_FREEMIUM

    @property
    def MAX_FREE_VOICE_NOTE_RECORDING_SECONDS(self) -> int:
        return EnvConfig.MAX_FREE_VOICE_NOTE_RECORDING_SECONDS

    @property
    def MAX_FREE_CREDITS(self) -> int:
        return EnvConfig.MAX_FREE_CREDITS

    @property
    def STATE_EXPIRATION_MINUTES(self) -> int:
        return EnvConfig.STATE_EXPIRATION_MINUTES

    @property
    def MAX_WA_MESSAGE_CHAR_LENGTH(self) -> int:
        return EnvConfig.MAX_WA_MESSAGE_CHAR_LENGTH

    @property
    def NEXT_TRANSCRIPTION_SEPARATOR(self) -> str:
        return EnvConfig.NEXT_TRANSCRIPTION_SEPARATOR

    @property
    def RATES_UNLIMITED_ADMIN(self) -> str:
        return EnvConfig._RATES_UNLIMITED_ADMIN

    @property
    def PAYG_PER_MINUTE_OF_TRANSCRIPTION_GBP(self) -> float:
        return EnvConfig._PAYG_PER_MINUTE_OF_TRANSCRIPTION_GBP

    @property
    def AUDIO_FILE_VOICENOTE_EXPECTED_FORMAT(self) -> Literal['ogg']:
        return EnvConfig.AUDIO_FILE_VOICENOTE_EXPECTED_FORMAT

    @property
    def RATES_PAYG(self) -> str:
        return EnvConfig.RATES_PAYG

    @property
    def RATES_UNLIMITED(self) -> str:
        return EnvConfig.RATES_UNLIMITED

    @property
    def TWILIO_ACCOUNT_SID(self) -> str:
        return EnvConfig.TWILIO_ACCOUNT_SID

    @property
    def TWILIO_AUTH_TOKEN(self) -> str:
        return EnvConfig.TWILIO_AUTH_TOKEN

    @property
    def OUR_TWILIO_NUMBER(self) -> str:
        return EnvConfig.OUR_TWILIO_NUMBER

    @property
    def OUR_TWILIO_SANDBOX_NUMBER(self) -> str:
        return EnvConfig.OUR_TWILIO_SANDBOX_NUMBER

    @property
    def ADMIN_PHONE(self) -> str:
        return EnvConfig.ADMIN_PHONE

    @property
    def STRIPE_PRIV_KEY_LIVE(self) -> str:
        return EnvConfig.STRIPE_PRIV_KEY_LIVE

    @property
    def STRIPE_PRIV_KEY_SANDBOX(self) -> str:
        return EnvConfig.STRIPE_PRIV_KEY_SANDBOX

    @property
    def USE_STRIPE_SANDBOX(self) -> bool:
        return EnvConfig.USE_STRIPE_SANDBOX

    @property
    def OPENAI_API_KEY(self) -> str:
        return EnvConfig.OPENAI_API_KEY

    @property
    def AI_MODEL(self) -> str:
        return EnvConfig.AI_MODEL

    @property
    def AI_IMAGE_MODEL(self) -> str:
        return EnvConfig.AI_IMAGE_MODEL

    @property
    def AI_TRANSCRIPTION_MODEL(self) -> str:
        return EnvConfig.AI_TRANSCRIPTION_MODEL

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return EnvConfig.ANTHROPIC_API_KEY

    @property
    def SESSION_WINDOW_TIMEDELTA(self) -> timedelta:
        return EnvConfig.SESSION_WINDOW_TIMEDELTA

    @property
    def GOOGLE_CLOUD_PUBLIC_DIR(self) -> str:
        return EnvConfig.GOOGLE_CLOUD_PUBLIC_DIR

    @property
    def GCS_BUCKET_NAME(self) -> str:
        return EnvConfig.GCS_BUCKET_NAME

    @property
    def GIPHY_API_KEY(self) -> str:
        return EnvConfig.GIPHY_API_KEY

    @property
    def REMINDER_FORMAT(self):
        return EnvConfig.REMINDER_FORMAT

    @property
    def EMAIL_IS_WHOLE_BODY_REGEX(self) -> str:
        return EnvConfig.EMAIL_IS_WHOLE_BODY_REGEX

    @property
    def EMAIL_IS_IN_BODY_REGEX(self) -> str:
        return EnvConfig.EMAIL_IS_IN_BODY_REGEX

    @property
    def REGEX_CONFIG(self):
        return EnvConfig.REGEX_CONFIG

    @property
    def price_per_token(self) -> dict:
        return EnvConfig.price_per_token

    @property
    def LOG_LEVEL(self) -> str:
        return EnvConfig.LOG_LEVEL

    @property
    def LOG_TO_FILE(self) -> bool:
        return EnvConfig.LOG_TO_FILE

    @property
    def LOG_TO_CONSOLE(self) -> bool:
        return EnvConfig.LOG_TO_CONSOLE

    @property
    def LOG_TO_REMOTE(self) -> bool:
        return EnvConfig.LOG_TO_REMOTE

    @property
    def REMOTE_LOGGING_URL(self) -> str:
        return EnvConfig.REMOTE_LOGGING_URL

    @property
    def SUPPORTED_LANGUAGES(self) -> list[str]:
        return EnvConfig.SUPPORTED_LANGUAGES

    @property
    def LOG_LEVEL_INITIAL(self) -> int:
        return EnvConfig.LOG_LEVEL_INITIAL

    @property
    def USE_POSTGRES_DB(self) -> bool:
        return EnvConfig.USE_POSTGRES_DB

    @property
    def LOGS_URL(self) -> str:
        return EnvConfig.LOGS_URL


Config = ReadOnlyConfig()
