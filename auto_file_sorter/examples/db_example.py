from auto_file_sorter.db.gmail_db import GmailDatabase


def main():
    # Initialize database
    with GmailDatabase() as db:
        # Create labels
        label_id = db.create_label("Important")
        print(f"Created label with ID: {label_id}")

        # Create a rule
        rule_data = {
            "conditions": {
                "from": "important@example.com",
                "subject": ".*urgent.*"
            },
            "actions": [
                {"type": "label", "value": "Important"},
                {"type": "star"}
            ]
        }
        rule_id = db.create_rule("Important Emails", rule_data)
        print(f"Created rule with ID: {rule_id}")

        # Get all labels
        labels = db.get_all_labels()
        print("All labels:", labels)

        # Get all rules
        rules = db.get_all_rules()
        print("All rules:", rules)


if __name__ == "__main__":
    main()
