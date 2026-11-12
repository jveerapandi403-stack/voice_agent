import os
import unittest
from email.message import EmailMessage
from unittest.mock import MagicMock, patch

from voice_agent import send_customer_enquiry


class SendCustomerEnquiryTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "BUSINESS_EMAIL": "business@example.com",
            "SMTP_EMAIL": "sender@example.com",
            "SMTP_PASSWORD": "test-password",
        },
        clear=True,
    )
    @patch("voice_agent.email_send.smtplib.SMTP_SSL")
    def test_sends_enquiry_using_gmail_smtp(
        self,
        smtp_ssl: MagicMock,
    ) -> None:
        server = smtp_ssl.return_value.__enter__.return_value

        send_customer_enquiry(
            {
                "customer_name": "Alex",
                "customer_phone": "555-0100",
                "reason_for_call": "Service question",
                "customer_requirement": "A repair",
                "preferred_date_time": "Friday morning",
                "additional_notes": "Call back after 10",
            }
        )

        smtp_ssl.assert_called_once_with(
            "smtp.gmail.com",
            465,
            timeout=15,
        )
        server.login.assert_called_once_with("sender@example.com", "test-password")
        sent_message = server.send_message.call_args.args[0]
        self.assertIsInstance(sent_message, EmailMessage)
        self.assertEqual(sent_message["To"], "business@example.com")
        self.assertEqual(sent_message["From"], "sender@example.com")
        self.assertEqual(sent_message["Subject"], "New Customer Enquiry - Alex")
        self.assertIn("Phone: 555-0100", sent_message.get_content())

    @patch.dict(os.environ, {}, clear=True)
    def test_reports_missing_email_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "Email configuration is missing"):
            send_customer_enquiry({})


if __name__ == "__main__":
    unittest.main()
