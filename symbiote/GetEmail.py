#!/usr/bin/env python3
#
# GetEmail.py

import imaplib
import poplib
import email
from email.parser import BytesParser
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import time
import re
from ollama import Client
olclient = Client(host='http://localhost:11434')
log("Loading symbiote GetEmail.")

class MailChecker:
    def __init__(self, username, password, mail_type='imap', days=None, unread=False, model=None):
        self.username = username
        self.password = password
        self.mail_type = mail_type.lower()
        self.days = days
        self.unread = unread
        self.imap_server = 'imap.gmail.com'
        self.pop_server = 'pop.gmail.com'
        self.model = model

    def check_mail(self):
        if self.mail_type == 'imap':
            return self._check_imap_mail()
        elif self.mail_type == 'pop':
            return self._check_pop_mail()
        else:
            raise ValueError("Unsupported mail type. Use 'imap' or 'pop'.")

    def _check_imap_mail(self):
        with imaplib.IMAP4_SSL(self.imap_server) as mail:
            mail.login(self.username, self.password)
            mail.select('inbox', readonly=True)  # Open mailbox in read-only mode

            search_criteria = []
            if self.unread:
                search_criteria.append('UNSEEN')
            if self.days is not None:
                date = (datetime.now() - timedelta(days=self.days)).strftime("%d-%b-%Y")
                search_criteria.append(f'SINCE {date}')
            
            search_criteria = ' '.join(search_criteria) or 'ALL'
            result, data = mail.search(None, search_criteria)
            mail_ids = data[0].split()

            emails = []
            count = 0
            for mail_id in mail_ids:
                count += 1
                log(f"Processing {count} of {len(mail_ids)} e-mails.")
                result, msg_data = mail.fetch(mail_id, '(RFC822)')
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                emails.append(self._get_email_content(msg))

            return emails

    def _check_pop_mail(self):
        with poplib.POP3_SSL(self.pop_server) as mail:
            mail.user(self.username)
            mail.pass_(self.password)
            num_messages = len(mail.list()[1])

            emails = []
            date_limit = datetime.now() - timedelta(days=self.days) if self.days is not None else None

            for i in range(num_messages):
                response, lines, octets = mail.retr(i + 1)
                raw_email = b"\n".join(lines)
                msg = BytesParser().parsebytes(raw_email)
                
                email_date = email.utils.parsedate_to_datetime(msg['Date'])
                if date_limit and email_date < date_limit:
                    continue
                
                emails.append(self._get_email_content(msg))

            return emails

    def _get_email_content(self, msg):
        from_ = msg.get('From', '').strip()
        to_ = msg.get('To', '').strip()
        subject = msg.get('Subject', '').strip()
        date = msg.get('Date', '').strip()
        date_unix = int(time.mktime(email.utils.parsedate(date)))
        body = self._get_body(msg).strip()

        if self.model is not None: 
            # Summarize e-mail body
            log(f"Summarizing e-mail body")
            system_prompt = """You are an expert in summarizing emails. Your task is to transform the body of an email into a concise, structured summary. This summary should accurately capture the key details and essence of the original email and include a summary of any attachments. Follow these guidelines:

    Identify the Purpose: Determine the primary purpose of the email (e.g., request, update, notification, inquiry, etc.).

    Extract Key Information: Include essential details such as names, dates, actions required, and any other critical information.

    Maintain Objectivity: Present the information without adding personal interpretations or opinions.

    Structure the Summary:

    Subject: [Brief description of the main topic]
    Summary: [Detailed summary capturing the main points and context of the email]
    Key Details: [Highlight specific details such as names, dates, deadlines, locations, etc.]
    Action Items (if applicable): [List of actions required, responsibilities, and deadlines]
    Attachment Summary: If any attachments are present, provide a brief summary of each attachment, including the type, content, and key details or purpose of the document.

    Length: The summary should be concise yet comprehensive, extracting and detailing all important points from the email body and attachments.

    Use this format for all email summaries, ensuring clarity and precision while thoroughly capturing the email's content and attachments."""
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": body})
            if self.model.startswith("ollama:"):
                model_name = self.model.split(":")
                use_model = model_name[1] + ":" + model_name[2]

                response = olclient.chat(
                        model=use_model,
                        messages=messages,
                        #options={ "num_ctx": 8192 },
                        stream=False,
                        )

                body = response['message']['content']
        
        email_content = {
            "from": from_,
            "to": to_,
            "subject": subject,
            "date": date_unix,
            "body": body[:10000]
        }
        
        return email_content

    def _get_body(self, msg):
        body = str() 
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if "attachment" not in content_disposition:
                    if content_type == "text/plain":
                        body += part.get_payload(decode=True).decode('utf-8', errors='replace')
                    elif content_type == "text/html":
                        html = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        body += self._extract_text_from_html(html)
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
            elif content_type == "text/html":
                html = msg.get_payload(decode=True).decode('utf-8', errors='replace')
                body = self._extract_text_from_html(html)
        
        # Clean up the body by removing redundant whitespace and zero-width characters
        body = self._clean_body(body)
        return body

    def _extract_text_from_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')

        # Remove style, script, head, and other irrelevant tags
        for element in soup(['style', 'script', 'head', 'title', 'meta', '[document]']):
            element.decompose()

        return soup.get_text(separator="\n", strip=True)

    def _clean_body(self, body):
        # Remove multiple consecutive newlines and redundant whitespace
        body = re.sub(r'\s+', ' ', body).strip()
        # Remove zero-width characters
        body = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', body)
        return body

if __name__ == "__main__":
    # Example usage:
    # Create an instance of the class
    mail_checker = MailChecker(
        username='USERNAME',
        password='PASSWORD',
        mail_type='imap',  # or 'pop'
        days=7,  # Check emails from the last 7 days
        unread=False,  # Check only unread emails
        model='phi3.5',
    )

    # Check emails
    emails = mail_checker.check_mail()

    # Print the JSON array of emails
    print(emails)

