import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import datetime as dt
import time

def sendemail(emailto,
              subject,
              emailfrom="wmlakefund.noreply@gmail.com",
              fileToSend=None,
              username="wmlakefund.noreply@gmail.com",
              password="wnxdtjhkdtromarz"):
    msg = MIMEMultipart()
    msg["From"] = emailfrom
    msg["To"] = emailto
    msg["Subject"] = subject
    msg.preamble = subject

    ctype, encoding = mimetypes.guess_type(fileToSend)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)

    if maintype == "text":
        fp = open(fileToSend)
        # Note: we should handle calculating the charset
        attachment = MIMEText(fp.read(), _subtype=subtype)
        fp.close()
    elif maintype == "image":
        fp = open(fileToSend, "rb")
        attachment = MIMEImage(fp.read(), _subtype=subtype)
        fp.close()
    elif maintype == "audio":
        fp = open(fileToSend, "rb")
        attachment = MIMEAudio(fp.read(), _subtype=subtype)
        fp.close()
    else:
        fp = open(fileToSend, "rb")
        attachment = MIMEBase(maintype, subtype)
        attachment.set_payload(fp.read())
        fp.close()
        encoders.encode_base64(attachment)

    fileNameShort = fileToSend
    if '\\' in fileToSend:
        fileNameShort = fileToSend.split('\\')[-1]
    else:
        fileNameShort = fileToSend.split('/')[-1]

    attachment.add_header("Content-Disposition", "attachment", filename=fileNameShort)
    msg.attach(attachment)
    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login(username, password)
    server.sendmail(emailfrom, emailto, msg.as_string())
    server.quit()


def sendemails(emailtolist, subject, fileToSend):
    for emailto in emailtolist:
        sendemail(emailto=emailto, subject=subject, fileToSend=fileToSend)


# convert date or datetime to seconds, as one argument for v5 API
def dt_to_millsec(date, dateformat='%Y-%m-%d'):
    if len(date) == 19 and dateformat == '%Y-%m-%d':
        dateformat = '%Y-%m-%d %H:%M:%S'
    t = dt.datetime.strptime(date, dateformat)
    return int(time.mktime(t.timetuple()) * 1000)
