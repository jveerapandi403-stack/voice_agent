import os
import smtplib
from email.message import EmailMessage
from typing import Mapping

SMTP_TIMEOUT_SECONDS = 15


def send_customer_enquiry(enquiry: Mapping[str, object]) -> dict[str, object]:
    business_email = os.getenv("BUSINESS_EMAIL")
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not business_email or not smtp_email or not smtp_password:
        raise ValueError("Email configuration is missing in .env")

    customer_name = enquiry.get("customer_name", "Unknown")
    message = EmailMessage()
    message["Subject"] = f"New Customer Enquiry - {customer_name}"
    message["From"] = smtp_email
    message["To"] = business_email
    message.set_content(
        f"""Customer Enquiry
Name: {enquiry.get('customer_name', 'Not provided')}
Phone: {enquiry.get('customer_phone', 'Not provided')}
Reason for calling: {enquiry.get('reason_for_call', 'Not provided')}
Requirement: {enquiry.get('customer_requirement', 'Not provided')}
Preferred date/time: {enquiry.get('preferred_date_time', 'Not provided')}
Additional notes: {enquiry.get('additional_notes', 'None')}"""
    )

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        timeout=SMTP_TIMEOUT_SECONDS,
    ) as server:
        server.login(smtp_email, smtp_password)
        server.send_message(message)

    return {"success": True, "message": "Customer enquiry email sent successfully."}
