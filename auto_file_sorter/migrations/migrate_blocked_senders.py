import re
from pathlib import Path
import logging
from auto_file_sorter.db.gmail_db import GmailDatabase


def determine_pattern_type(pattern: str) -> str:
    """Determine the type of blocking pattern"""
    if '@' in pattern and not any(c in pattern for c in '*?[](){}'):
        return 'email'
    elif pattern.startswith('body:'):
        return 'body_pattern'
    else:
        return 'pattern'


def migrate_blocked_senders(blocked_file_path: str, db: GmailDatabase) -> None:
    """Migrate blocked senders from file to database"""
    blocked_file = Path(blocked_file_path)
    if not blocked_file.exists():
        logging.warning(f"Blocked senders file not found: {blocked_file}")
        return

    # Create backup of original file
    backup_path = blocked_file.with_suffix('.txt.bak')
    blocked_file.rename(backup_path)
    logging.info(f"Created backup of blocked senders file: {backup_path}")

    # Read patterns from file and add to database
    with open(backup_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pattern_type = determine_pattern_type(line)
                # Remove 'body:' prefix if present
                if pattern_type == 'body_pattern':
                    line = line.replace('body:', '', 1).strip()

                rule_id = db.create_blocked_sender(line, pattern_type)
                if rule_id:
                    logging.info(f"Migrated blocked pattern: {
                                 line} ({pattern_type})")
                else:
                    logging.error(f"Failed to migrate pattern: {line}")


def main():
    logging.basicConfig(level=logging.INFO)
    db = GmailDatabase()

    blocked_file_path = '/Users/joey/Library/Mobile Documents/iCloud~is~workflow~my~workflows/Documents/blocked_senders.txt'
    migrate_blocked_senders(blocked_file_path, db)

    db.close()


if __name__ == "__main__":
    main()
