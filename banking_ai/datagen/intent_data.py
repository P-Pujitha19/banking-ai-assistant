"""Synthetic training data for the TensorFlow intent classifier (templates x slot fillers x noise)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

INTENTS = [
    "balance_inquiry", "transaction_history", "recent_transaction", "loan_eligibility", "loan_information",
    "interest_rate", "account_information", "account_charges", "minimum_balance", "emi_calculation",
    "transaction_explanation", "loan_comparison", "greeting", "goodbye", "help", "fallback",
]

SLOTS = {
    "amount": ["50000", "2 lakh", "3 lakh", "5 lakh", "7.5 lakh", "10 lakh", "₹3,00,000", "75k", "25 lakh", "1 crore", "rs 8,00,000", "four lakh"],
    "tenure": ["2 years", "3 years", "5 years", "7 years", "10 years", "15 years", "20 years", "36 months", "60 months", "5 yrs"],
    "n": ["2", "3", "5", "7", "10", "15", "20"],
    "loan": ["personal loan", "home loan", "car loan", "education loan", "housing loan", "auto loan", "student loan"],
    "loan2": ["home loan", "car loan", "personal loan", "education loan"],
    "acct": ["savings", "current", "salary"],
    "rate": ["8.5%", "10%", "11.5%", "9 percent", "12%", "7.9%"],
    "income": ["40000", "60k", "85,000", "1.2 lakh", "50000"],
    "txn": ["TXN10000221", "TXN10000456", "txn10001002"],
    "month": ["august", "july", "september"],
    "period": ["this month", "last month", "last week", "yesterday", "last 30 days", "last 3 months", "this week"],
}

TEMPLATES: dict[str, list[str]] = {
    "balance_inquiry": [
        "what is my account balance", "show my balance", "how much money do i have", "check balance", "what's my available balance",
        "available balance please", "how much is in my account", "tell me my current balance", "balance enquiry",
        "how much money is left in my {acct} account", "what is the balance in my account", "can you check my balance",
        "i want to know my balance", "what is my ledger balance", "how much funds do i have", "show me the balance of my {acct} account",
        "remaining balance in my account", "what is my total balance", "current balance of my account", "how much do i have in the bank",
        "what is the available balance in my account", "account balance", "my balance", "balance please", "what is my balance",
    ],
    "transaction_history": [
        "show my recent transactions", "show my last {n} transactions", "list my transactions", "show transactions from {period}",
        "show my transactions for {period}", "transaction history", "mini statement", "give me my account statement",
        "show my debit transactions", "show all credits {period}", "what are my transactions in the {period}", "show my upi transactions",
        "display my last {n} transactions", "get my past {n} transactions", "i want to see my transaction history", "show my atm withdrawals",
        "transactions between 2026-08-01 and 2026-08-31", "view my statement for {period}", "show all my debits", "show my recent activity",
        "list the last {n} transactions on my account", "show my transactions in {month}", "what transactions happened {period}",
        "show recent payments from my account", "i need my account statement", "show my neft transactions",
    ],
    "recent_transaction": [
        "what was my last transaction", "show my latest transaction", "what is my most recent transaction", "last transaction details",
        "tell me about my last payment", "what was the last debit", "what was my last credit", "when was my last transaction",
        "my last transaction please", "what was the most recent payment from my account", "latest transaction", "what about my last transaction",
        "and my last transaction", "last debit from my account", "what was the last amount credited", "what is the last transaction on my account",
        "how much was my last transaction", "tell me my previous transaction", "last upi payment", "when did i last make a payment",
    ],
    "loan_eligibility": [
        "can i get a {loan}", "am i eligible for a {loan}", "check my loan eligibility", "can i get a {loan} of {amount}",
        "can i get a {loan} of {amount} for {tenure}", "i need a loan of {amount} for {tenure}", "i want a {loan}", "i need {amount}",
        "i want to take a {loan}", "i need a {loan} of {amount}", "can i afford this loan", "can i afford a loan of {amount}",
        "will my {loan} of {amount} be approved", "eligibility for {loan}", "how much loan can i get", "what is the maximum loan i can get",
        "do i qualify for a {loan}", "i earn {income} per month can i get a {loan}", "my salary is {income} am i eligible for a {loan} of {amount}",
        "i would like to apply for a {loan}", "is a loan of {amount} affordable for me", "need a loan of {amount}", "i want to borrow {amount}",
        "how much {loan} am i eligible for", "i am looking for a {loan}", "check if i can get {amount} loan", "am i eligible for {amount} {loan} for {tenure}",
        "i need money", "will i get a {loan}", "can i afford the emi for a {loan} of {amount}",
    ],
    "loan_information": [
        "what loans are available", "which loans do you offer", "tell me about your loan products", "what types of loans do you have",
        "tell me about the {loan}", "details of {loan}", "what is the tenure of a {loan}", "what documents are required for a {loan}",
        "what is the processing fee for a {loan}", "what is the maximum amount for a {loan}", "do you offer {loan}", "loan products list",
        "i want to know about loans", "features of {loan}", "how do i apply for a {loan}", "prepayment charges on {loan}",
        "what is the age limit for {loan}", "show me all loans", "what is a {loan}", "loan options", "tell me about loans",
    ],
    "interest_rate": [
        "what is the interest rate for a {loan}", "{loan} interest rate", "what is the current interest rate", "current rates on savings account",
        "fixed deposit interest rates", "what rate do you charge on {loan}", "how much interest will i pay on a {loan}", "interest rates please",
        "rate of interest for {loan}", "what interest do i get on my savings account", "what are your loan rates", "tell me the roi on {loan}",
        "how much interest does my savings account earn", "latest interest rates", "what is the rd interest rate", "what is the fd rate for 1 year",
        "{loan} rates", "interest rate", "what are the current home loan rates", "what rate will i get on a {loan}",
    ],
    "account_information": [
        "tell me about my savings account", "show my account details", "what type of account do i have", "give me my account information",
        "what are the features of my account", "account summary", "details of my {acct} account", "what accounts do i have with you",
        "when did i open my account", "what is my account status", "tell me about my {acct} account", "my account number", "show my profile",
        "account type and status", "what is my daily upi limit", "what is the atm withdrawal limit", "transaction limits on my account",
        "what benefits does my account have", "what is my ifsc code", "which branch is my account in", "explain the features of a {acct} account",
        "how many accounts do i have", "what is the daily transfer limit",
    ],
    "account_charges": [
        "what are the charges for my account", "what fees do you charge on my account", "what are the atm charges", "service charges on my account",
        "how much do you charge for imps", "account maintenance charges", "debit card annual fee", "sms alert charges", "what are the bank charges",
        "any hidden charges on my account", "charges on {acct} account", "what is the fee for neft", "how much is the cheque book fee", "list of all fees",
        "what is the gst on charges", "how much will it cost to get a duplicate statement", "show me the fee schedule", "what charges have been applied recently",
        "what are the account charges", "banking fees",
    ],
    "minimum_balance": [
        "what is the minimum balance requirement", "minimum balance for savings account", "how much minimum balance should i maintain",
        "what happens if i don't maintain minimum balance", "average monthly balance required", "minimum balance in current account",
        "what is the penalty for low balance", "min balance", "do i need to maintain a minimum balance", "what is the minimum average balance for {acct} account",
        "minimum balance penalty", "how much balance should i keep in my {acct} account", "what is the minimum balance", "is there a minimum balance requirement",
        "zero balance account", "what is the non maintenance charge",
    ],
    "emi_calculation": [
        "calculate my emi", "calculate emi for {amount} for {tenure} at {rate}", "what will be the emi for a {loan} of {amount}",
        "emi for {amount} over {tenure}", "emi calculator", "how much emi will i pay for {amount} for {tenure}",
        "monthly instalment for a loan of {amount} for {tenure} at {rate}", "compute emi", "what is my monthly emi for {tenure}",
        "emi on {amount} at {rate} for {tenure}", "what would the emi be for {amount}", "work out the monthly payment on a {loan} of {amount}",
        "tell me the emi for {tenure}", "calculate the emi", "what is the emi for a {loan} of {amount} for {tenure}", "monthly payment for {amount} at {rate}",
        "find emi for {amount} loan", "how much will i pay every month for {amount} over {tenure}",
    ],
    "transaction_explanation": [
        "why was money deducted from my account", "why was this amount debited", "what is this deduction", "explain this transaction",
        "why did i get charged", "what was the charge in my account", "unknown debit in my account", "why was {amount} deducted",
        "i see a deduction i don't recognise", "explain transaction {txn}", "what is this charge on my statement", "why is there a debit of {amount}",
        "why was my balance reduced", "what is gst charge in my statement", "why was i charged a penalty", "what is this sms alert charge",
        "what is txn {txn} for", "why was a fee taken from my account", "who is this payment to", "explain the deduction",
    ],
    "loan_comparison": [
        "compare personal loan and home loan", "{loan} vs {loan2}", "difference between {loan} and {loan2}", "which is better {loan} or {loan2}",
        "compare {loan} with {loan2}", "compare loans", "compare the loan products", "how is a home loan different from a personal loan",
        "home loan versus car loan", "{loan} or {loan2} which should i choose", "compare {loan} and {loan2} interest rates", "personal loan vs education loan",
        "what is the difference between a {loan} and a {loan2}",
    ],
    "greeting": ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "hello there", "hi assistant", "namaste", "hey there", "hi there", "hello assistant", "hii", "hey bot"],
    "goodbye": ["bye", "goodbye", "thank you", "thanks", "thanks that's all", "see you later", "that will be all", "ok thanks bye", "no more questions", "have a nice day",
                "cheers", "thanks a lot", "thank you so much", "bye bye", "that's all for now", "ok thanks"],
    "help": ["help", "what can you do", "how can you help me", "what services do you provide", "i need help", "what can i ask you", "show me what you can do",
             "options", "how does this work", "list your features", "what are your capabilities", "help me", "what do you do"],
    "fallback": [
        "what's the weather today", "tell me a joke", "who won the cricket match", "recommend a good stock to buy", "what is the capital of france",
        "book a flight to delhi", "play some music", "how do i cook biryani", "asdfgh qwerty", "blah blah", "what is bitcoin price",
        "who is the prime minister", "write a poem", "translate hello into french", "i am bored", "what time is it in london", "order me a pizza",
        "how tall is mount everest", "tell me about movies", "zzzz", "what should i wear today", "open the pod bay doors", "what is the meaning of life",
        "what about {tenure}", "and for {tenure}", "make it {tenure}", "for {tenure} then", "{amount}", "what if {amount}", "ok {amount}",
        "what about {amount}", "how about {tenure}", "{tenure}", "and {rate}", "yes", "no", "ok", "maybe", "what about that", "and then",
        "can you hack the bank", "give me free money", "who created you", "are you a human", "do you like pizza",
    ],
}

PREFIXES = ["", "", "", "", "hi, ", "hello, ", "please ", "hey, ", "quick question - ", "i want to know, ", "can you tell me ", "could you tell me "]
SUFFIXES = ["", "", "", "", " please", " thanks", " ?", " today", " now"]
SHORT_INTENTS = {"greeting", "goodbye", "help"}


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for key, values in SLOTS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(values), 1)
    return out


def _noise(text: str, intent: str, rng: random.Random) -> str:
    if intent in SHORT_INTENTS:
        return text + rng.choice(["", "", "", "!", " :)"])
    prefix = rng.choice(PREFIXES) if intent != "fallback" or len(text.split()) > 3 else ""
    if text.startswith(("can i", "am i", "do i", "will", "what", "how", "why", "is ", "which")) and prefix.startswith(("can you", "could you")):
        prefix = ""
    out = prefix + text + rng.choice(SUFFIXES)
    roll = rng.random()
    if roll < 0.08:
        out = out.upper()
    elif roll < 0.14:
        out = out.title()
    if rng.random() < 0.12:
        out = out.replace("'", "")
    return out.strip()


def generate_intent_dataset(seed: int = 7, per_intent: int = 150) -> list[tuple[str, str]]:
    """Return ``(text, intent)`` rows (de-duplicated, shuffled)."""
    rng = random.Random(seed)
    rows: set[tuple[str, str]] = set()
    for intent, templates in TEMPLATES.items():
        generated: set[str] = set()
        attempts = 0
        while len(generated) < per_intent and attempts < per_intent * 30:
            attempts += 1
            template = templates[attempts % len(templates)] if attempts <= len(templates) else rng.choice(templates)
            text = _noise(_fill(template, rng), intent, rng)
            generated.add(text)
        rows |= {(t, intent) for t in generated}
    data = sorted(rows)
    rng.shuffle(data)
    return data


def write_intent_dataset(path: Path, seed: int = 7, per_intent: int = 150) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_intent_dataset(seed, per_intent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)
    return len(rows)
