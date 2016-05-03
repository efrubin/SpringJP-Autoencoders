import smtplib
SERVER = "localhost"
FROM = "e@erubin.xyz"
TO = ["erubin@princeton.edu"]

SUBJECT = "Your Python Script has Finished Execution"
TEXT = "pretty dice"
message = """ \
From: %s
To: %s
Subject %s

%s
""" % (FROM, ", ".join(TO), SUBJECT, TEXT)

server = smtplib.SMTP(SERVER)
server.sendmail(FROM, TO, message)
server.quit()
