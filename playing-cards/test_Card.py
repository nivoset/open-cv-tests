import unittest
from unittest.mock import patch, MagicMock
from Hand import Hand

class TestCard(unittest.TestCase):
    @patch('Card.YOLO')  # Mock the YOLO class
    @patch('builtins.open')  # Mock open to read the YAML file
    @patch('yaml.safe_load')  # Mock yaml.safe_load to return a fake configuration
    def test_init(self, mock_safe_load, mock_open, mock_yolo):
        # Setup mock responses
        mock_safe_load.return_value = {'names': ['Ace', 'King']}
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance

        # Test data
        settings = {
            'device': 'cpu',
            'confidence_threshold': 0.5,
            'debug': False
        }

        # Create an instance of Card
        card = Hand(settings)
        
        # Assertions to verify correct initialization
        self.assertEqual(card.device, 'cpu')
        self.assertEqual(card.confidence_threshold, 0.5)
        self.assertEqual(card.debug, False)
        mock_yolo.assert_called_once_with("playing-cards/playing-card-model/weights/best.pt")

    @patch('Card.YOLO')
    def test_format_hand(self, mock_yolo):
        # Mock configuration setup
        mock_yolo.return_value = None  # We don't need the actual YOLO object for this test
        config = {'names': ['Ace', 'King']}
        card = Hand({'device': 'cpu', 'confidence_threshold': 0.5, 'debug': False})
        card.settings = {'names': ['Ace', 'King']}

        # Sample hand data
        hand = [
            MagicMock(cls=[0], conf=[MagicMock(item=MagicMock(return_value=0.95))]),
            MagicMock(cls=[1], conf=[MagicMock(item=MagicMock(return_value=0.88))])
        ]

        formatted_hand = card.format_hand(hand)

        # Check if the hand is formatted correctly
        self.assertEqual(formatted_hand, [('ACE', 0.95), ('KING', 0.88)])

    @patch('Card.YOLO')
    def test_from_hand(self, mock_yolo):
        # Setup
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance
        mock_yolo_instance.return_value = MagicMock(boxes=['box1', 'box2'])

        card = Hand({'device': 'cpu', 'confidence_threshold': 0.5, 'debug': False})
        card.format_hand = MagicMock(return_value='formatted_hand')

        # Test data
        card_images = ['image1', 'image2']
        
        # Execute
        hands = card.from_hand(card_images)

        # Verify
        self.assertEqual(hands, ['formatted_hand', 'formatted_hand'])
        card.format_hand.assert_called()

if __name__ == '__main__':
    unittest.main()
