import os
import shutil
from pathlib import Path
from typing import Dict, List
import mimetypes
import PIL
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import re


class DownloadsManager:
    def __init__(self, downloads_path: str = "/Users/joey/Downloads"):
        self.downloads_path = Path(downloads_path)
        self.extension_mappings = {
            # Documents
            'pdf': 'documents/pdfs',
            'doc': 'documents',
            'docx': 'documents',
            'txt': 'documents',
            'rtf': 'documents',

            # Presentations
            'ppt': 'presentations',
            'pptx': 'presentations',
            'key': 'presentations',

            # Spreadsheets
            'xls': 'spreadsheets',
            'xlsx': 'spreadsheets',
            'csv': 'spreadsheets',

            # Images
            'jpg': 'images',
            'jpeg': 'images',
            'png': 'images',
            'gif': 'images',
            'svg': 'images',

            # Audio
            'mp3': 'audio',
            'wav': 'audio',
            'm4a': 'audio',
            'flac': 'audio',

            # Video
            'mp4': 'video',
            'mov': 'video',
            'avi': 'video',
            'mkv': 'video',

            # Archives
            'zip': 'archives',
            'rar': 'archives',
            '7z': 'archives',
            'tar': 'archives',
            'gz': 'archives'
        }

        # Initialize image classification model
        self.image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/resnet-50")
        self.image_model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50")

    def create_directory_structure(self):
        """Create the necessary subdirectories if they don't exist"""
        unique_paths = set(self.extension_mappings.values())
        for path in unique_paths:
            full_path = self.downloads_path / path
            full_path.mkdir(parents=True, exist_ok=True)

    def analyze_image(self, image_path: Path) -> str:
        """Analyze image content and return a descriptive filename"""
        try:
            image = Image.open(image_path)
            inputs = self.image_processor(image, return_tensors="pt")
            outputs = self.image_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_label = self.image_model.config.id2label[probs.argmax(
            ).item()]

            # Clean up the label to make it filename-friendly
            clean_label = re.sub(r'[^\w\s-]', '', pred_label).strip().lower()
            clean_label = re.sub(r'[-\s]+', '-', clean_label)

            return clean_label
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return image_path.stem

    def get_target_directory(self, file_path: Path) -> Path:
        """Determine the target directory based on file extension"""
        extension = file_path.suffix.lower().lstrip('.')
        relative_path = file_path.relative_to(self.downloads_path)
        current_subpath = relative_path.parent if relative_path.parent != Path(
            '.') else Path('')

        if extension in self.extension_mappings:
            new_subpath = self.extension_mappings[extension]
            return self.downloads_path / current_subpath / new_subpath
        return self.downloads_path / current_subpath

    def sanitize_filename(self, filename: str) -> str:
        """Clean up filename to remove invalid characters and make it more descriptive"""
        # Remove invalid characters
        clean_name = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        clean_name = clean_name.replace(' ', '_')
        # Remove multiple underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        return clean_name

    def rename_ambiguous_files(self):
        """Rename files with ambiguous names"""
        for file_path in self.downloads_path.rglob('*'):
            if file_path.is_file():
                # Skip already organized files
                if any(subdir in str(file_path) for subdir in set(self.extension_mappings.values())):
                    continue

                original_name = file_path.stem
                extension = file_path.suffix.lower()

                # Check if filename is ambiguous (e.g., "download", "untitled", etc.)
                ambiguous_patterns = ['download',
                                      'untitled', 'image', 'doc', 'file']
                if any(pattern in original_name.lower() for pattern in ambiguous_patterns):
                    new_name = original_name

                    # For images, use AI to analyze content
                    if extension.lstrip('.') in ['jpg', 'jpeg', 'png', 'gif']:
                        new_name = self.analyze_image(file_path)

                    # Add timestamp to ensure uniqueness
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"{new_name}_{timestamp}{extension}"
                    new_filepath = file_path.parent / new_filename

                    try:
                        file_path.rename(new_filepath)
                        print(f"Renamed: {file_path.name} -> {new_filename}")
                    except Exception as e:
                        print(f"Error renaming {file_path}: {str(e)}")

    def organize_files(self):
        """Organize files into appropriate subdirectories"""
        # First create necessary directories
        self.create_directory_structure()

        # Then rename ambiguous files
        self.rename_ambiguous_files()

        # Finally, move files to appropriate directories
        for file_path in self.downloads_path.rglob('*'):
            if file_path.is_file():
                # Skip already organized files
                if any(subdir in str(file_path) for subdir in set(self.extension_mappings.values())):
                    continue

                target_dir = self.get_target_directory(file_path)
                if not target_dir.exists():
                    target_dir.mkdir(parents=True, exist_ok=True)

                target_path = target_dir / file_path.name

                # Handle filename conflicts
                if target_path.exists():
                    base_name = target_path.stem
                    extension = target_path.suffix
                    counter = 1
                    while target_path.exists():
                        new_name = f"{base_name}_{counter}{extension}"
                        target_path = target_dir / new_name
                        counter += 1

                try:
                    shutil.move(str(file_path), str(target_path))
                    print(f"Moved: {file_path} -> {target_path}")
                except Exception as e:
                    print(f"Error moving {file_path}: {str(e)}")


def main():
    # Initialize and run the downloads manager
    manager = DownloadsManager()
    manager.organize_files()


if __name__ == "__main__":
    main()
