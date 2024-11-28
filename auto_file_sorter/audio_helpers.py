from typing import TYPE_CHECKING
from typing import Literal, TypeGuard
from pydub import AudioSegment
import tempfile

# import wave
import audioread
import logging
import os

import httpx
from config import Config
from utils import format_time_to_mins_from_secs
import hashlib

type AudioFileAcceptedFormat = Literal['wav', 'mp3',
                                       'flac', 'ogg', 'mp4', 'm4a', 'wave', 'CAF']


def get_file_hash(file_path: str, hash_algorithm=hashlib.md5) -> str:
    """
    Calculate the hash of a file.

    :param file_path: Path to the file
    :param hash_algorithm: Hashing algorithm to use (default is MD5)
    :return: Hexadecimal digest of the file hash
    """
    hash_object = hash_algorithm()
    with open(file_path, "rb") as file:
        # Read and update hash in chunks of 4K
        for chunk in iter(lambda: file.read(4096), b""):
            hash_object.update(chunk)
    return hash_object.hexdigest()


def get_audio_duration(file_path, format: str):
    """
    This Python function calculates the duration of an audio file in seconds given its file path.

    :param file_path: The `file_path` parameter in the `get_audio_duration` function should be the path
    to the audio file for which you want to determine the duration. This should be a string representing
    the file path on your system where the audio file is located. For example, it could be something
    like "C
    :return: The function `get_audio_duration` returns the duration of the audio file in seconds.
    """
    if not audio_file_has_supported_format(format):
        raise ValueError(
            f"Unsupported file format: {format}, needs to be one of .wav, .mp3, .flac, .ogg, mp4, .m4a, .wave, CAF")
    audio = AudioSegment.from_file(file_path, format=format)
    duration_in_milliseconds = len(audio)
    duration_in_seconds = duration_in_milliseconds / 1000
    return duration_in_seconds


def verify_audio(file_path):
    try:
        with audioread.audio_open(file_path) as audio_file:
            logging.debug(
                f"verify_audio() -> Duration: {audio_file.duration} seconds")
            return True
    except FileNotFoundError:
        logging.error(f"verify_audio() -> Audio file not found 🟡: {file_path}")
        return False
    except Exception as e:
        logging.error(f"verify_audio() -> Error reading audio file 🟡: {
                      file_path} with error: \n{e}", exc_info=True)


def verify_file_is_ogg(file_path: str):
    """
    The function `verify_file_is_ogg` checks if a file is in OGG format.

    :param file_path: Path to the audio file to check
    :type file_path: str
    :return: Boolean indicating if the file is a valid OGG file
    """
    if not file_path.lower().endswith('.ogg'):
        return False
    try:
        return True
        # Check MIME type
        # import filetype
        # file_type = filetype.filetype(file_path)
        # return file_type.mime == 'audio/ogg'
    except Exception as e:
        logging.error(f"verify_file_is_ogg() -> Error reading audio file 🟡: {
                      file_path} with error: \n{e}", exc_info=True)
        return False


async def get_media_type_from_url(media_url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.head(media_url)
        content_type = response.headers.get('content-type', '')
        return content_type.split('/')[0]


def get_media_type_from_url_sync(media_url: str) -> str:
    return httpx.head(media_url).headers.get('content-type', '').split('/')[0]


def get_http_request_content_type(response: httpx.Response) -> tuple[Literal['vcard', 'audio', 'video', 'image', ''], str]:
    content_type_header = response.headers.get(
        'Content-Type', 'audio/ogg').split('/')
    media_type = f"{content_type_header[0]}"
    file_ext = f"{content_type_header[1]}"

    if media_type == 'text' and file_ext == 'vcard':
        media_type = 'vcard'
        file_ext = 'vcf'

    assert media_type in ('vcard', 'audio', 'video',
                          'image'), f"Expected a media_type of 'audio', 'video' or 'image' but got {media_type}"
    return media_type, file_ext


def trim_audio_to_audio_segment(file_path: str):
    assert isinstance(file_path, str), f"trim_audio_to_audio_segment expected a file_path of type: [str] but received a [{
        type(file_path)}] type from file_path: \"{file_path}\"."
    if not os.path.exists(
            file_path):
        raise FileNotFoundError(
            f"trim_audio_to_audio_segment expected a file_path that exists: \"{file_path}\".")
    assert file_path.endswith(Config.AUDIO_FILE_VOICENOTE_EXPECTED_FORMAT), f"trim_audio_to_audio_segment expected a file_path of format: [{
        Config.AUDIO_FILE_VOICENOTE_EXPECTED_FORMAT}] but received a [{file_path}] file type from file_path: \"{file_path}\"."
    try:
        voice_note_file_format = Config.AUDIO_FILE_VOICENOTE_EXPECTED_FORMAT
        root, sep_ext = os.path.splitext(p=file_path)
        ext = sep_ext.split(".")[-1]
        # new_file_path = file_path
        assert ext == voice_note_file_format, f"trim_audio_to_new_file expected a file_path of format: [{
            voice_note_file_format}] but received a [{sep_ext}] file type from file_path: \"{file_path}\"."
        audio = AudioSegment.from_file(
            file_path, format=voice_note_file_format)
        assert isinstance(
            audio, AudioSegment), "audio of format ogg should create an AudioSegment object using AudioSegment.from_file(file_path, format=\"ogg\")"
        # audio = AudioSegment.from_file_using_temporary_files(file_path, format="ogg")

        # a = AudioSegment.from_mp3(mp3file)
        # first_second = a[:1000] # get the first second of an mp3
        # slice = a[5000:10000] # get a slice from 5 to 10 seconds of an mp3
        duration_in_milliseconds = len(audio)
        half_duration_in_millis = duration_in_milliseconds // 2
        # Trim to at least 30s of audio or full audio if less than 30s, max 60s and if audio is between 30s and 2mins long, then trim to half.
        trim_to = min(duration_in_milliseconds, max(half_duration_in_millis,
                      Config.MIN_FREEMIUM_AUDIO_TRIM * 1000), Config.MAX_FREEMIUM_AUDIO_TRIM * 1000)
        if trim_to < duration_in_milliseconds:
            # FREEMIUM TRIM:
            # if not new_file_path:
            #     new_file_path = f"{root}_trimmed.{ext}"
            logging.debug(f"FREEMIUM AUDIO TRIMMING from {format_time_to_mins_from_secs(duration_in_milliseconds//1000)}s of audio to {format_time_to_mins_from_secs(trim_to//1000)}s using rule:\n\t" +
                          f"Trim to at least {format_time_to_mins_from_secs(Config.MIN_FREEMIUM_AUDIO_TRIM)}s of audio or full audio if less than {format_time_to_mins_from_secs(Config.MIN_FREEMIUM_AUDIO_TRIM)}s, max {format_time_to_mins_from_secs(Config.MAX_FREEMIUM_AUDIO_TRIM)}s and if audio is between {format_time_to_mins_from_secs(Config.MIN_FREEMIUM_AUDIO_TRIM)}s and {format_time_to_mins_from_secs(Config.MAX_FREEMIUM_AUDIO_TRIM * 2)}s long, then trim to half")
            trimmed_audio_segment = audio[:trim_to]
            assert isinstance(trimmed_audio_segment, AudioSegment), f"trimmed_audio_segment of format ogg should be an AudioSegment object using AudioSegment.from_file(file_path, format=\"ogg\")[:{
                trim_to}], but we got a [{type(trimmed_audio_segment).__name__}] object"

            # new_file_path = False
            # if new_file_path:
            #     trimmed_audio_segment.export(
            #         out_f=new_file_path, format=voice_note_file_format)
            # # return new_file_path
            # else:
            #     if isinstance(trimmed_audio_segment.raw_data, bytes):
            #         new_temp_file_wrapper.write(
            #             content=trimmed_audio_segment.raw_data.decode("utf-8"))
            #     else:
            #         logging.error(f"Unable to get raw_data bytes out of the trimmed AudioSegment instance from the {file_path} file that we trimmed from {
            #                       format_time_to_mins_from_secs(duration_in_milliseconds//1000)}s of audio to {format_time_to_mins_from_secs(trim_to//1000)}s")
            #         return None
            return trimmed_audio_segment
        else:
            # return file_path
            return audio
    except FileNotFoundError:
        logging.error(f"trim_audio() -> Audio file not found 🟡: {file_path}")
        return None
    except Exception as e:
        logging.error(f"trim_audio() -> Error reading audio file 🟡: {
                      file_path} with error: \n{e}", exc_info=True)
        return None


def audio_file_has_supported_extension(file_path: str) -> bool:
    return file_path.lower().endswith(('wav', 'mp3', 'flac', 'ogg', 'mp4', 'm4a', 'wave', 'CAF'))


def audio_file_has_supported_format(format: str) -> TypeGuard[AudioFileAcceptedFormat]:
    return format.lower() in ('ogg', 'mp4', 'm4a', 'wav', 'CAF', 'mp3', 'flac')


def AudioFileAcceptedFormatTypeGuard(format: str) -> TypeGuard[AudioFileAcceptedFormat]:
    return format.lower() in ('ogg', 'mp4', 'm4a', 'wav', 'CAF', 'mp3', 'flac')


def convert_audio_file_to_ogg(format: AudioFileAcceptedFormat | None, file_path: str, format_to: Literal['ogg'] = 'ogg') -> str:
    temp_file = None
    try:
        # Handle MP4 conversion if needed
        ext = format.lower() if format else file_path.split(".")[-1].lower()
        out_file_path = file_path.replace(ext, format_to)
        if ext == "ogg":
            return file_path
        elif ext == "mp4":
            # Create temporary file for converted audio
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f'.{format_to}', delete=False)
            audio = AudioSegment.from_file(file_path, format=ext)
            audio.export(out_file_path, format=format_to)
            file_path = out_file_path
        else:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f'.{format_to}', delete=False)
            audio = AudioSegment.from_file(file_path, format=ext)
            audio.export(out_file_path, format=format_to)
            file_path = out_file_path
    finally:
        # Clean up temporary file if it was created
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
    return file_path


async def download_media(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.content


def download_media_sync(url: str) -> bytes:
    return httpx.get(url).content
