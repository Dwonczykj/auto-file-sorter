import os
import logging
import tempfile


# This class likely represents a temporary file that can be used for storing data temporarily.
class TemporaryFileFromBytes:
    def __init__(self, content: bytes, file_extension: str):
        """
        The above function is a Python constructor that initializes an object with a file path
        attribute.

        :param file_path: The `__init__` method you provided is a constructor for a class. It
        initializes an instance of the class with a `file_path` attribute. The `file_path` parameter is
        the path to a file that the instance will work with or represent
        """
        self._content = content
        if not file_extension.startswith("."):
            file_extension = f".{file_extension}"
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(content)
            self.temp_file_path = temp_file.name
            self._temp_file = temp_file

    def __enter__(self):
        return self.temp_file_path

    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)


class TemporaryFileFromStr:
    def __init__(self, content: str, file_extension: str):
        """
        The above function is a Python constructor that initializes an object with a file path
        attribute.

        :param file_path: The `__init__` method you provided is a constructor for a class. It
        initializes an instance of the class with a `file_path` attribute. The `file_path` parameter is
        the path to a file that the instance will work with or represent
        """
        self._content = content
        if not file_extension.startswith("."):
            file_extension = f".{file_extension}"
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(content.encode('utf-8'))
            self.temp_file_path = temp_file.name
            self._temp_file = temp_file

    def __enter__(self):
        return self.temp_file_path

    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)


class TemporaryFileForWriting:

    def __init__(self, file_extension: str):
        """
        The above function is a Python constructor that initializes an object with a file path
        attribute.

        :param file_path: The `__init__` method you provided is a constructor for a class. It
        initializes an instance of the class with a `file_path` attribute. The `file_path` parameter is
        the path to a file that the instance will work with or represent
        """
        # self._content = content
        if not file_extension.startswith("."):
            file_extension = f".{file_extension}"
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            # temp_file.write(content.encode('utf-8'))
            self.temp_file_path = temp_file.name
            self._temp_file = temp_file

    def __enter__(self):
        return self._temp_file

    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)
