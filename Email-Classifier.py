import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

data = pd.read_csv('/home/aicha/spam.csv', encoding='latin-1')

X = data['Message']
y = data['Category']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

classifier = MultinomialNB()
classifier.fit(X, y)

window = tk.Tk()
window.title("Email Spam Classifier")
def classify_emails():
    email_text = email_entry.get("1.0", "end-1c")
    if email_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter an email!")
        return
    
    email_transformed = vectorizer.transform([email_text])
    prediction = classifier.predict(email_transformed)[0]
    
    if prediction == 'spam':
        result_label.config(text="Spam", fg="red")
    else:
        result_label.config(text="Not Spam", fg="green")


email_label = tk.Label(window, text="Email:")
email_label.pack()

email_entry = tk.Text(window, height=10, width=50)
email_entry.pack()

classify_button = tk.Button(window, text="Classify", command=classify_emails)
classify_button.pack()

result_label = tk.Label(window, text="")
result_label.pack()

window.mainloop()
