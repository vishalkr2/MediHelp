import json
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackHandler:
    def __init__(self, feedback_file='feedback_data.json'):
        self.feedback_file = os.path.join(os.path.dirname(__file__), feedback_file)
        self.load_feedback_data()

    def load_feedback_data(self):
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                self.feedback_data = json.load(f)
            logger.info("Feedback data loaded successfully.")
        else:
            self.feedback_data = []
            logger.info("No existing feedback data found. Starting fresh.")

    def save_feedback_data(self):
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=4)
        logger.info("Feedback data saved successfully.")

    def add_feedback(self, query, response, is_helpful, user_comment=None):
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'is_helpful': is_helpful,
            'user_comment': user_comment
        }
        self.feedback_data.append(feedback)
        logger.debug(f"Feedback data before saving: {self.feedback_data}")
        self.save_feedback_data()
        logger.info(f"Feedback added: {feedback}")

    def get_feedback_stats(self):
        if not self.feedback_data:
            logger.info("No feedback data available.")
            return {'helpful': 0, 'not_helpful': 0, 'total': 0}

        helpful = sum(1 for f in self.feedback_data if f['is_helpful'])
        total = len(self.feedback_data)
        
        logger.info(f"Feedback stats - Total: {total}, Helpful: {helpful}, Not Helpful: {total - helpful}")
        return {
            'helpful': helpful,
            'not_helpful': total - helpful,
            'total': total
        }
