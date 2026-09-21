"""
💰 Loan Advisor - Home Loan | Car Loan | Appliance & Gadget Loan
------------------------------------------------------------------
A simple Streamlit app with a separate advisor for each type of loan.

🏠 HOME LOAN
   EMI calculator, eligibility (FOIR + LTV + credit score + age),
   repayment schedule, prepayment simulator, bank comparison,
   income-tax benefits, AI advice

🚗 CAR LOAN
   EMI + true cost of owning the car, eligibility, the 20/4/10 smart-buying
   rule, loan balance vs car value (depreciation), prepayment,
   bank comparison, AI advice

📱 APPLIANCE / GADGET LOAN (TV, phone, laptop, fridge, AC ...)
   EMI + processing fee, eligibility, smart-buying checks (product life),
   "No-Cost EMI" checker (finds the hidden interest), tenure comparison,
   repayment schedule, AI advice

📊 OVERALL SUMMARY
   What your total EMIs would look like if you took all the loans.

This project is intentionally kept simple (plain functions, no classes,
no advanced patterns) so it is easy to explain in a college presentation.

Author: Student Project (RE-03)
"""

import math

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from google import genai

# ---------------------------------------------------------
# SETTINGS / CONSTANTS  (change them here in one place)
# ---------------------------------------------------------
GEMINI_MODEL = "gemini-3.6-flash"   # if the API says "model not found", change this name

TAX_LIMIT_80C = 150000              # Section 80C limit on principal repayment (per year)
TAX_LIMIT_24B = 200000              # Section 24(b) limit on interest (self-occupied house)
RETIREMENT_AGE = 60                 # banks want a home loan closed by this age
CAR_MAX_AGE = 65                    # age by which a car loan should be closed
CAR_MAX_FINANCE = 90                # banks finance up to ~90% of the car price (approx.)

# Typical useful life (in years) of common products - used to warn the
# user if the loan lasts too long compared to how long the product lasts.
PRODUCT_LIFE_YEARS = {
    "Smartphone": 3,
    "Laptop": 4,
    "Television": 8,
    "Refrigerator": 10,
    "Washing Machine": 8,
    "Air Conditioner": 8,
    "Other": 5,
}

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Loan Advisor", page_icon="💰", layout="wide")


# ---------------------------------------------------------
# SMALL HELPER FUNCTIONS
# ---------------------------------------------------------

def format_inr(amount):
    """Format a number in the Indian style: 2500000 -> '₹ 25,00,000'."""
    amount = round(amount)
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))

    if len(digits) <= 3:
        return f"{sign}₹ {digits}"

    last_three = digits[-3:]
    rest = digits[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)

    return f"{sign}₹ {','.join(parts)},{last_three}"


def format_tenure(months):
    """Turn a number of months into text: 245 -> '20 yrs 5 months'."""
    months = int(months)
    years = months // 12
    remaining = months % 12
    if years == 0:
        return f"{remaining} months"
    return f"{years} yrs {remaining} months"


# ---------------------------------------------------------
# CORE FINANCE FUNCTIONS (shared by all three loan types)
# Plain Python maths - no external library needed, so it is
# easy to explain to a class. Everything works in MONTHS
# internally, so it suits both a 20-year home loan and a
# 6-month phone loan.
# ---------------------------------------------------------

def calculate_emi_months(principal, annual_rate, months):
    """
    Calculate the Equated Monthly Installment (EMI).

    Formula:
        EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)

    where:
        P = loan amount (principal)
        r = monthly interest rate (annual rate / 12 / 100)
        n = number of monthly installments
    """
    monthly_rate = annual_rate / (12 * 100)

    if monthly_rate == 0:  # edge case: 0% interest loan
        return principal / months

    emi = principal * monthly_rate * (1 + monthly_rate) ** months
    emi = emi / ((1 + monthly_rate) ** months - 1)
    return emi


def calculate_emi(principal, annual_rate, tenure_years):
    """Same as calculate_emi_months, but the tenure is given in years."""
    return calculate_emi_months(principal, annual_rate, tenure_years * 12)


def generate_schedule_months(principal, annual_rate, months):
    """
    Build a month-wise repayment schedule showing how much of each
    EMI goes towards interest vs principal, and the remaining balance.
    Returns a pandas DataFrame.
    """
    months = int(months)
    monthly_rate = annual_rate / (12 * 100)
    emi = calculate_emi_months(principal, annual_rate, months)

    balance = principal
    rows = []
    for month in range(1, months + 1):
        interest_payment = balance * monthly_rate
        principal_payment = emi - interest_payment
        balance = balance - principal_payment
        if balance < 0:
            balance = 0

        rows.append({
            "Month": month,
            "EMI": round(emi, 2),
            "Principal Paid": round(principal_payment, 2),
            "Interest Paid": round(interest_payment, 2),
            "Remaining Balance": round(balance, 2)
        })

    return pd.DataFrame(rows)


def generate_amortization_schedule(principal, annual_rate, tenure_years):
    """Same as generate_schedule_months, but the tenure is given in years."""
    return generate_schedule_months(principal, annual_rate, tenure_years * 12)


def generate_yearly_summary(schedule_df):
    """
    Group the monthly schedule into a year-wise summary
    (total principal paid, total interest paid, balance at year end).
    """
    df = schedule_df.copy()
    df["Year"] = (df["Month"] - 1) // 12 + 1

    yearly = df.groupby("Year").agg({
        "Principal Paid": "sum",
        "Interest Paid": "sum",
        "Remaining Balance": "last"
    }).reset_index()

    return yearly.round(2)


def check_eligibility_months(monthly_income, existing_emis, requested_loan,
                             annual_rate, months, foir_limit=50):
    """
    A simplified loan-eligibility check based on FOIR
    (Fixed Obligation to Income Ratio) - a common real-world rule
    used by lenders: total EMIs (old + new) should not exceed a
    certain percentage (commonly 40-50%) of monthly income.

    Returns a dictionary with the eligibility result and details.
    """
    max_allowed_emi = (foir_limit / 100) * monthly_income
    emi_available_for_new_loan = max_allowed_emi - existing_emis

    requested_emi = calculate_emi_months(requested_loan, annual_rate, months)

    is_eligible = requested_emi <= emi_available_for_new_loan and emi_available_for_new_loan > 0

    # Also estimate the maximum loan amount the person could get
    # with the remaining EMI budget (reverse EMI formula)
    monthly_rate = annual_rate / (12 * 100)
    if emi_available_for_new_loan > 0 and monthly_rate > 0:
        max_eligible_loan = emi_available_for_new_loan * ((1 + monthly_rate) ** months - 1)
        max_eligible_loan = max_eligible_loan / (monthly_rate * (1 + monthly_rate) ** months)
    elif emi_available_for_new_loan > 0:
        max_eligible_loan = emi_available_for_new_loan * months
    else:
        max_eligible_loan = 0

    return {
        "is_eligible": is_eligible,
        "requested_emi": round(requested_emi, 2),
        "max_allowed_emi": round(max_allowed_emi, 2),
        "emi_budget_left": round(emi_available_for_new_loan, 2),
        "max_eligible_loan": round(max_eligible_loan, 2)
    }


def check_eligibility(monthly_income, existing_emis, requested_loan,
                      annual_rate, tenure_years, foir_limit=50):
    """Same as check_eligibility_months, but the tenure is given in years."""
    return check_eligibility_months(monthly_income, existing_emis, requested_loan,
                                    annual_rate, tenure_years * 12, foir_limit)


def simulate_prepayment(principal, annual_rate, tenure_years,
                        extra_monthly=0, lump_sum=0, lump_sum_month=12):
    """
    Simulate paying extra money towards the loan.
    The EMI stays the same, but the extra payment reduces the principal
    faster, so the loan finishes early and less interest is paid.

    extra_monthly  = extra amount paid every month along with the EMI
    lump_sum       = one-time extra payment
    lump_sum_month = the month number in which the lump sum is paid
    """
    monthly_rate = annual_rate / (12 * 100)
    max_months = tenure_years * 12
    emi = calculate_emi(principal, annual_rate, tenure_years)

    balance = principal
    total_interest = 0
    month = 0
    balances = []

    while balance > 0.5 and month < max_months:
        month += 1
        interest_payment = balance * monthly_rate
        principal_payment = emi - interest_payment

        extra = extra_monthly
        if month == lump_sum_month:
            extra = extra + lump_sum

        balance = balance - principal_payment - extra
        if balance < 0:
            balance = 0

        total_interest = total_interest + interest_payment
        balances.append(balance)

    return {
        "months_taken": month,
        "total_interest": total_interest,
        "balances": balances
    }


def compare_banks(bank_df, principal, tenure_years):
    """
    Calculate EMI and total interest for every bank/rate in a table,
    so the user can compare offers side by side.
    """
    rows = []
    for _, row in bank_df.iterrows():
        bank = row["Bank"]
        rate = row["Interest Rate (%)"]

        # skip empty / invalid rows the user may have added
        if pd.isna(bank) or pd.isna(rate) or rate <= 0:
            continue

        emi = calculate_emi(principal, rate, tenure_years)
        total_payment = emi * tenure_years * 12

        rows.append({
            "Bank": bank,
            "Interest Rate (%)": rate,
            "Monthly EMI (₹)": round(emi),
            "Total Interest (₹)": round(total_payment - principal),
            "Total Payment (₹)": round(total_payment)
        })

    return pd.DataFrame(rows)


def rate_sensitivity(principal, base_rate, tenure_years,
                     changes=(-1, -0.5, 0, 0.5, 1, 2)):
    """
    "What if the interest rate changes?" - shows how the EMI and
    total interest change if the rate goes up or down.
    """
    base_emi = calculate_emi(principal, base_rate, tenure_years)
    rows = []
    for change in changes:
        rate = round(base_rate + change, 2)
        if rate <= 0:
            continue
        emi = calculate_emi(principal, rate, tenure_years)
        rows.append({
            "Interest Rate (%)": rate,
            "Change": f"{change:+.1f}%",
            "Monthly EMI (₹)": round(emi),
            "EMI Change (₹)": round(emi - base_emi),
            "Total Interest (₹)": round(emi * tenure_years * 12 - principal)
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# ELIGIBILITY CHECK BUILDERS (shared by all three loan types)
# Every check is a small dictionary with a status:
#   "pass" = all good, "warn" = be careful, "fail" = problem
# ---------------------------------------------------------

def make_check(name, status, detail):
    """Create one check result."""
    return {"name": name, "status": status, "detail": detail}


def foir_check(foir):
    """Check 1 (all loans): can the person afford the new EMI?"""
    if foir["is_eligible"]:
        return make_check(
            "EMI affordability", "pass",
            f"New EMI {format_inr(foir['requested_emi'])} fits within your "
            f"EMI budget of {format_inr(foir['emi_budget_left'])}.")
    if foir["emi_budget_left"] <= 0:
        return make_check("EMI affordability", "fail",
                          "Your existing EMIs already use up your full EMI limit.")
    return make_check(
        "EMI affordability", "fail",
        f"New EMI {format_inr(foir['requested_emi'])} is more than your "
        f"EMI budget of {format_inr(foir['emi_budget_left'])}.")


def credit_score_check(credit_score):
    """Check 2 (all loans): the credit (CIBIL) score."""
    if credit_score >= 750:
        return make_check("Credit score", "pass",
                          f"{credit_score} is excellent - you can expect the best interest rates.")
    if credit_score >= 700:
        return make_check("Credit score", "pass",
                          f"{credit_score} is good - most lenders will approve the loan.")
    if credit_score >= 650:
        return make_check("Credit score", "warn",
                          f"{credit_score} is fair - approval is possible but the rate may be higher.")
    return make_check("Credit score", "fail",
                      f"{credit_score} is low - most lenders are likely to reject the application.")


def age_check(age, tenure_years, max_age, min_age=21):
    """Check 3 (all loans): is the person's age suitable for this tenure?"""
    if age < min_age:
        return make_check("Age", "fail",
                          f"Most lenders need the applicant to be at least {min_age} years old.")

    if age + tenure_years > max_age:
        max_tenure = max_age - age
        if max_tenure > 0:
            detail = (f"The loan would end after age {max_age}. Lenders may reduce "
                      f"the tenure to about {format_tenure(max_tenure * 12)}.")
        else:
            detail = "Lenders may ask for a younger co-applicant at your age."
        return make_check("Age vs tenure", "warn", detail)

    return make_check("Age vs tenure", "pass",
                      f"The loan will be fully repaid by age {math.ceil(age + tenure_years)}.")


def get_max_ltv(loan_amount):
    """
    Maximum Loan-to-Value (LTV) ratio allowed for a HOME loan, based on
    the loan size (simplified version of the RBI guidelines):
        loan up to 30 lakh       -> up to 90% of property value
        loan 30 lakh to 75 lakh  -> up to 80%
        loan above 75 lakh       -> up to 75%
    """
    if loan_amount <= 3000000:
        return 90
    elif loan_amount <= 7500000:
        return 80
    else:
        return 75


def home_ltv_check(loan_amount, property_value):
    """Home loan only: is the loan too large compared to the property value?"""
    ltv = loan_amount / property_value * 100
    max_ltv = get_max_ltv(loan_amount)
    min_down_payment = property_value * (1 - max_ltv / 100)

    if ltv <= max_ltv:
        return make_check(
            "Loan-to-Value (LTV)", "pass",
            f"Loan is {ltv:.1f}% of the property value (limit {max_ltv}%). "
            f"Down payment needed: {format_inr(property_value - loan_amount)}.")
    return make_check(
        "Loan-to-Value (LTV)", "fail",
        f"Loan is {ltv:.1f}% of the property value, above the {max_ltv}% limit. "
        f"You need a down payment of at least {format_inr(min_down_payment)}.")


def car_financing_check(car_price, car_loan):
    """Car loan only: banks do not finance 100% of the car price."""
    financed_pct = car_loan / car_price * 100
    min_down_payment = car_price * (1 - CAR_MAX_FINANCE / 100)

    if financed_pct <= CAR_MAX_FINANCE:
        return make_check(
            "Loan vs car price", "pass",
            f"Loan covers {financed_pct:.1f}% of the car price (banks usually allow "
            f"up to about {CAR_MAX_FINANCE}%). Down payment: {format_inr(car_price - car_loan)}.")
    return make_check(
        "Loan vs car price", "fail",
        f"Loan covers {financed_pct:.1f}% of the car price, above the usual "
        f"{CAR_MAX_FINANCE}% limit. Pay at least {format_inr(min_down_payment)} upfront.")


def finish_checks(checks, foir):
    """The loan is 'likely eligible' only if no check has failed."""
    overall = all(c["status"] != "fail" for c in checks)
    return {"overall": overall, "checks": checks, "foir": foir}


# ---------------------------------------------------------
# HOME LOAN FUNCTIONS
# ---------------------------------------------------------

def run_home_checks(monthly_income, existing_emis, loan_amount, annual_rate,
                    tenure_years, age, credit_score, property_value):
    """Run all eligibility checks for a HOME loan."""
    foir = check_eligibility(monthly_income, existing_emis, loan_amount,
                             annual_rate, tenure_years)
    checks = [foir_check(foir)]
    if property_value > 0:
        checks.append(home_ltv_check(loan_amount, property_value))
    checks.append(credit_score_check(credit_score))
    checks.append(age_check(age, tenure_years, RETIREMENT_AGE))
    return finish_checks(checks, foir)


def calculate_tax_benefit(yearly_df, tax_slab):
    """
    Estimate the yearly income-tax saving on a home loan
    (India, OLD tax regime, self-occupied house):
        Section 80C   -> principal repaid, up to Rs 1.5 lakh per year
        Section 24(b) -> interest paid, up to Rs 2 lakh per year
    tax_slab is the user's tax rate in percent (e.g. 20 or 30).
    """
    rows = []
    for _, row in yearly_df.iterrows():
        deduction_80c = min(row["Principal Paid"], TAX_LIMIT_80C)
        deduction_24b = min(row["Interest Paid"], TAX_LIMIT_24B)
        tax_saved = (deduction_80c + deduction_24b) * tax_slab / 100

        rows.append({
            "Year": int(row["Year"]),
            "Principal Paid": round(row["Principal Paid"]),
            "Interest Paid": round(row["Interest Paid"]),
            "80C Deduction": round(deduction_80c),
            "24(b) Deduction": round(deduction_24b),
            "Tax Saved": round(tax_saved)
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# CAR LOAN FUNCTIONS
# ---------------------------------------------------------

def run_car_checks(monthly_income, existing_emis, car_loan, car_price,
                   annual_rate, tenure_years, age, credit_score):
    """Run all eligibility checks for a CAR loan."""
    foir = check_eligibility(monthly_income, existing_emis, car_loan,
                             annual_rate, tenure_years)
    checks = [
        foir_check(foir),
        car_financing_check(car_price, car_loan),
        credit_score_check(credit_score),
        age_check(age, tenure_years, CAR_MAX_AGE),
    ]
    return finish_checks(checks, foir)


def car_smart_rule_checks(car_price, down_payment, tenure_years, emi,
                          running_cost, monthly_income):
    """
    The popular "20/4/10 rule" for buying a car sensibly:
        20% -> pay at least 20% of the price as down payment
        4   -> take the loan for at most 4 years
        10% -> total monthly car cost (EMI + fuel + insurance + maintenance)
               should be at most 10% of your monthly income
    These are guidelines (not bank rules), so they only give warnings.
    """
    checks = []

    down_pct = down_payment / car_price * 100
    if down_pct >= 20:
        checks.append(make_check("20% down payment", "pass",
                                 f"You are paying {down_pct:.0f}% upfront. Great!"))
    else:
        checks.append(make_check("20% down payment", "warn",
                                 f"You are paying only {down_pct:.0f}% upfront. A bigger down "
                                 "payment means a smaller loan and less interest."))

    if tenure_years <= 4:
        checks.append(make_check("4-year tenure", "pass",
                                 f"A {tenure_years}-year loan keeps the interest cost low."))
    else:
        checks.append(make_check("4-year tenure", "warn",
                                 f"A {tenure_years}-year loan means more interest, and the car "
                                 "loses value faster than a long loan is repaid."))

    total_monthly = emi + running_cost
    if monthly_income > 0:
        cost_pct = total_monthly / monthly_income * 100
    else:
        cost_pct = 100
    if cost_pct <= 10:
        checks.append(make_check("10% income rule", "pass",
                                 f"Your car will cost {cost_pct:.1f}% of your monthly income."))
    else:
        checks.append(make_check("10% income rule", "warn",
                                 f"Your car will cost {cost_pct:.1f}% of your monthly income "
                                 "(guideline is 10%). Consider a cheaper car or a bigger down payment."))
    return checks


def estimate_car_values(car_price, months, yearly_depreciation):
    """
    Estimate the car's resale value every month, assuming it loses
    a fixed percentage of its value each year.
    """
    values = []
    for month in range(1, months + 1):
        value = car_price * (1 - yearly_depreciation / 100) ** (month / 12)
        values.append(value)
    return values


# ---------------------------------------------------------
# APPLIANCE / GADGET LOAN FUNCTIONS
# ---------------------------------------------------------

def run_gadget_checks(monthly_income, existing_emis, loan_amount,
                      annual_rate, months, age, credit_score):
    """Run all eligibility checks for a GADGET / APPLIANCE loan."""
    foir = check_eligibility_months(monthly_income, existing_emis, loan_amount,
                                    annual_rate, months)
    checks = [
        foir_check(foir),
        credit_score_check(credit_score),
        age_check(age, months / 12, CAR_MAX_AGE, min_age=18),
    ]
    return finish_checks(checks, foir)


def gadget_smart_checks(product, price, emi, months, extra_cost, monthly_income):
    """
    Smart-buying advice for gadgets (only warnings, never a 'fail'):
    1. The EMI should be a small part of the income (about 10% or less)
    2. The loan should end well before the product wears out
    3. Interest + fees should not be a big share of the price
    """
    checks = []

    if monthly_income > 0:
        emi_pct = emi / monthly_income * 100
    else:
        emi_pct = 100
    if emi_pct <= 10:
        checks.append(make_check("EMI vs income", "pass",
                                 f"The EMI is {emi_pct:.1f}% of your monthly income - comfortable."))
    else:
        checks.append(make_check("EMI vs income", "warn",
                                 f"The EMI is {emi_pct:.1f}% of your monthly income. "
                                 "Try to keep gadget EMIs under about 10%."))

    life_years = PRODUCT_LIFE_YEARS[product]
    tenure_years = months / 12
    if tenure_years <= life_years / 2:
        checks.append(make_check("Loan vs product life", "pass",
                                 f"A {product.lower()} typically lasts about {life_years} years. "
                                 "Your loan ends well before that."))
    else:
        checks.append(make_check("Loan vs product life", "warn",
                                 f"A {product.lower()} typically lasts about {life_years} years, "
                                 f"but you would be paying for {format_tenure(months)}. "
                                 "Try a shorter tenure."))

    extra_pct = extra_cost / price * 100
    if extra_pct <= 10:
        checks.append(make_check("Extra cost", "pass",
                                 f"Interest and fees add only {extra_pct:.1f}% to the price."))
    else:
        checks.append(make_check("Extra cost", "warn",
                                 f"Interest and fees add {extra_pct:.1f}% to the price "
                                 f"({format_inr(extra_cost)} extra). Compare with paying in full."))
    return checks


def calculate_effective_rate(cash_price, monthly_emi, months, upfront_fee=0):
    """
    Find the REAL yearly interest rate hidden in an EMI offer.

    The buyer "receives" the product worth cash_price (minus any upfront fee)
    and pays monthly_emi for 'months' months. We search (bisection method)
    for the monthly rate at which the EMIs are worth exactly that amount.
    Returns the yearly rate in percent (0 if the EMIs cost nothing extra).
    """
    amount_financed = cash_price - upfront_fee

    if amount_financed <= 0 or monthly_emi * months <= amount_financed:
        return 0.0

    low = 0.0    # monthly rate of 0%
    high = 1.0   # monthly rate of 100% (surely too high)
    for _ in range(100):
        mid = (low + high) / 2
        present_value = monthly_emi * (1 - (1 + mid) ** (-months)) / mid
        if present_value > amount_financed:
            low = mid     # the rate is too low, search higher
        else:
            high = mid    # the rate is too high, search lower

    monthly_rate = (low + high) / 2
    return monthly_rate * 12 * 100


def tenure_comparison(loan_amount, annual_rate, month_options):
    """Compare EMI and total interest for different loan tenures (in months)."""
    rows = []
    for months in month_options:
        emi = calculate_emi_months(loan_amount, annual_rate, months)
        rows.append({
            "Tenure (months)": months,
            "Monthly EMI (₹)": round(emi),
            "Total Interest (₹)": round(emi * months - loan_amount),
            "Total Payment (₹)": round(emi * months)
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# REPORT + AI FUNCTIONS
# ---------------------------------------------------------

def create_summary_report(title, details, results, check_result):
    """Build a plain-text summary the user can download."""
    lines = [title, "=" * len(title), "", "YOUR DETAILS"]
    for label, value in details:
        lines.append(f"{label:<24}: {value}")

    lines.append("")
    lines.append("RESULT")
    for label, value in results:
        lines.append(f"{label:<24}: {value}")
    lines.append(f"{'Likely Eligible':<24}: {'YES' if check_result['overall'] else 'NO'}")

    lines.append("")
    lines.append("ELIGIBILITY CHECKS")
    for check in check_result["checks"]:
        lines.append(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")

    lines.append("")
    lines.append("Note: This is an estimate for learning purposes, not financial advice.")
    return "\n".join(lines)


def checks_to_text(checks):
    """Turn a list of checks into text (used inside the AI prompt)."""
    return "\n".join(f"- {c['name']}: {c['status']} ({c['detail']})" for c in checks)


def ask_gemini_for_advice(api_key, loan_type, profile_summary, user_question=""):
    """
    Send the user's loan profile to Google Gemini and get back
    simple, friendly, easy-to-understand loan advice.

    Uses the current google-genai SDK (client.models.generate_content),
    not the older/deprecated google-generativeai package.

    loan_type is e.g. "home loan", "car loan" or "gadget loan", so the AI
    answers like a specialist for that kind of loan.
    """
    client = genai.Client(api_key=api_key)

    if user_question.strip():
        task = f"""
        The user has a specific question. Answer it first, in simple words:
        "{user_question.strip()}"
        """
    else:
        task = f"""
        Cover:
        - Whether the {loan_type} looks affordable for them
        - 2-3 tips to improve eligibility or reduce the interest burden
        - One general {loan_type} tip for beginners
        """

    prompt = f"""
    You are a friendly {loan_type} advisor talking to a first-time borrower.
    Based on the details below, give short, simple, practical advice
    (use plain language, avoid jargon, use bullet points, max 200 words).

    {profile_summary}

    {task}
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text


# ---------------------------------------------------------
# REUSABLE SCREEN PARTS
# (Small functions that draw a part of the page. The same part
#  is reused by different loan tabs. 'key' is a unique name that
#  Streamlit needs so that two widgets never clash.)
# ---------------------------------------------------------

def show_pie_chart(labels, values, title):
    """Draw a pie chart (slices with zero value are skipped)."""
    kept = [(label, value) for label, value in zip(labels, values) if value > 0]

    fig, ax = plt.subplots()
    ax.pie([value for _, value in kept], labels=[label for label, _ in kept],
           autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def show_emi_summary(loan_amount, emi, total_interest, total_payment):
    """Show the three EMI numbers and the principal-vs-interest pie chart."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly EMI", format_inr(emi))
    col2.metric("Total Interest Payable", format_inr(total_interest))
    col3.metric("Total Payment (Principal + Interest)", format_inr(total_payment))

    show_pie_chart(["Principal", "Total Interest"], [loan_amount, total_interest],
                   "Principal vs Interest")


def show_checks(checks):
    """Show a list of checks, each with an icon."""
    icons = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    for check in checks:
        st.write(f"{icons[check['status']]} **{check['name']}:** {check['detail']}")


def show_eligibility_result(check_result, existing_emis, emi, monthly_income):
    """Show the eligibility banner, all checks, the EMI-to-income bar and FOIR numbers."""
    if check_result["overall"]:
        st.success("✅ You are likely ELIGIBLE for this loan!")
    else:
        st.error("❌ You may NOT be eligible for this loan right now. See the failed checks below.")

    show_checks(check_result["checks"])
    st.markdown("---")

    if monthly_income > 0:
        ratio = (existing_emis + emi) / monthly_income * 100
    else:
        ratio = 100
    st.write(f"**Total EMIs as a share of your income: {ratio:.1f}%** (lender limit is about 50%)")
    st.progress(min(ratio / 100, 1.0))

    foir = check_result["foir"]
    colA, colB = st.columns(2)
    with colA:
        st.write(f"**Required EMI for this loan:** {format_inr(foir['requested_emi'])}")
        st.write(f"**Max EMI you can afford (50% rule):** {format_inr(foir['max_allowed_emi'])}")
    with colB:
        st.write(f"**EMI budget left (after existing EMIs):** {format_inr(foir['emi_budget_left'])}")
        st.write(f"**Maximum loan you may be eligible for:** {format_inr(foir['max_eligible_loan'])}")


def show_schedule_tab(loan_amount, rate, months, key, file_name, show_yearly=True):
    """Repayment schedule table + balance chart (+ yearly summary) + CSV download."""
    st.subheader("Month-by-Month Repayment Schedule")

    schedule_df = generate_schedule_months(loan_amount, rate, months)
    st.dataframe(schedule_df, use_container_width=True, height=350)

    fig, ax = plt.subplots()
    ax.plot(schedule_df["Month"], schedule_df["Remaining Balance"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Remaining Balance (₹)")
    ax.set_title("Loan Balance Over Time")
    st.pyplot(fig)
    plt.close(fig)

    if show_yearly:
        st.subheader("Year-by-Year Summary")
        yearly_df = generate_yearly_summary(schedule_df)
        st.dataframe(yearly_df, use_container_width=True)

        fig2, ax2 = plt.subplots()
        ax2.bar(yearly_df["Year"], yearly_df["Principal Paid"], label="Principal Paid")
        ax2.bar(yearly_df["Year"], yearly_df["Interest Paid"],
                bottom=yearly_df["Principal Paid"], label="Interest Paid")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Amount Paid (₹)")
        ax2.set_title("How Your Yearly Payments Are Split")
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption("In the early years most of your EMI goes towards interest. "
                   "Over time, more and more goes towards the principal.")

    csv = schedule_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Schedule as CSV", csv, file_name, "text/csv",
                       key=f"{key}_schedule_download")


def show_prepayment_tab(loan_amount, rate, tenure_years, key):
    """Prepayment simulator (used by home loan and car loan)."""
    st.subheader("Prepayment Simulator")
    st.write("See how paying some extra money can close your loan earlier and save interest.")

    total_months = tenure_years * 12
    colP1, colP2, colP3 = st.columns(3)
    extra_monthly = colP1.number_input("Extra payment every month (₹)", min_value=0,
                                       value=2000, step=500, key=f"{key}_extra")
    lump_sum = colP2.number_input("One-time lump sum payment (₹)", min_value=0,
                                  value=0, step=10000, key=f"{key}_lump")
    lump_sum_month = colP3.number_input("Lump sum paid in month no.", min_value=1,
                                        max_value=total_months,
                                        value=min(12, total_months), step=1,
                                        key=f"{key}_lump_month")

    original = simulate_prepayment(loan_amount, rate, tenure_years)
    with_prepay = simulate_prepayment(loan_amount, rate, tenure_years,
                                      extra_monthly, lump_sum, lump_sum_month)

    months_saved = original["months_taken"] - with_prepay["months_taken"]
    interest_saved = original["total_interest"] - with_prepay["total_interest"]

    m1, m2, m3 = st.columns(3)
    m1.metric("New Loan Tenure", format_tenure(with_prepay["months_taken"]))
    m2.metric("Time Saved", format_tenure(months_saved))
    m3.metric("Interest Saved", format_inr(interest_saved))

    if extra_monthly == 0 and lump_sum == 0:
        st.info("Enter an extra monthly amount or a lump sum to see the savings.")
    else:
        st.success(f"By prepaying, you finish {format_tenure(months_saved)} earlier "
                   f"and save {format_inr(interest_saved)} in interest.")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(original["balances"]) + 1), original["balances"],
            label="Without prepayment")
    ax.plot(range(1, len(with_prepay["balances"]) + 1), with_prepay["balances"],
            label="With prepayment")
    ax.set_xlabel("Month")
    ax.set_ylabel("Remaining Balance (₹)")
    ax.set_title("Loan Balance: With vs Without Prepayment")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def show_bank_comparison_tab(loan_amount, rate, tenure_years, sample_rates, key):
    """Bank comparison + rate sensitivity (used by home loan and car loan)."""
    st.subheader("Compare Banks / Interest Rates")
    st.write("Edit the table below (change the names and rates, or add rows) to compare offers "
             "for your loan amount and tenure. The rates shown are only sample values.")

    default_banks = pd.DataFrame({
        "Bank": ["Bank A", "Bank B", "Bank C"],
        "Interest Rate (%)": sample_rates
    })
    edited_banks = st.data_editor(default_banks, num_rows="dynamic", key=f"{key}_banks")

    compare_df = compare_banks(edited_banks, loan_amount, tenure_years)

    if compare_df.empty:
        st.info("Add at least one bank with an interest rate above 0.")
    else:
        st.dataframe(compare_df)

        best = compare_df.loc[compare_df["Total Interest (₹)"].idxmin()]
        worst = compare_df.loc[compare_df["Total Interest (₹)"].idxmax()]
        st.success(f"🏆 Cheapest option: **{best['Bank']}** at {best['Interest Rate (%)']}%.")
        if len(compare_df) > 1:
            saving = worst["Total Interest (₹)"] - best["Total Interest (₹)"]
            st.write(f"Choosing {best['Bank']} instead of {worst['Bank']} "
                     f"saves you **{format_inr(saving)}** over the full tenure.")

        st.bar_chart(compare_df.set_index("Bank")["Total Interest (₹)"])

    st.markdown("---")
    st.subheader("What If the Interest Rate Changes?")
    st.write(f"Your current rate is {rate}%. Rates can change, "
             "so here is how your EMI would change:")
    st.dataframe(rate_sensitivity(loan_amount, rate, tenure_years))


def show_ai_tab(loan_type, profile_summary, key):
    """AI advice box (used by all loan types). Uses the API key from the sidebar."""
    st.subheader(f"AI-Powered {loan_type.title()} Advice (Google Gemini)")

    user_question = st.text_input(
        "Have a specific question? (optional)",
        placeholder="e.g. Should I choose a shorter tenure?",
        key=f"{key}_question"
    )

    if st.button("Get AI Recommendation", key=f"{key}_ai_button"):
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar first.")
        else:
            with st.spinner("Asking Gemini for advice..."):
                try:
                    advice = ask_gemini_for_advice(api_key, loan_type, profile_summary,
                                                   user_question)
                    st.markdown(advice)
                except Exception as e:
                    st.error(f"Something went wrong while calling Gemini API: {e}")
    else:
        st.info("Click the button above to get personalized, easy-to-understand loan advice.")


# ---------------------------------------------------------
# SIDEBAR - API KEY + PERSONAL DETAILS (shared by all loans)
# ---------------------------------------------------------
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")
st.sidebar.caption("Get a free key from https://aistudio.google.com/app/apikey")

st.sidebar.markdown("---")
st.sidebar.subheader("Your Details")
age = st.sidebar.number_input("Your Age (years)", min_value=18, max_value=70, value=30, step=1)
monthly_income = st.sidebar.number_input("Monthly Income (₹)", min_value=0, value=60000, step=1000)
existing_emis = st.sidebar.number_input("Existing Monthly EMIs (₹)", min_value=0, value=0, step=500)
credit_score = st.sidebar.number_input("Credit (CIBIL) Score", min_value=300, max_value=900, value=750, step=10)
st.sidebar.caption("These details are used by all three loan advisors. "
                   "Loan-specific details are entered inside each tab.")


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------
st.title("💰 Loan Advisor")
st.write("Pick the type of loan you are planning. Each advisor has its own calculators, "
         "eligibility rules and tips.")

tab_home, tab_car, tab_gadget, tab_summary = st.tabs([
    "🏠 Home Loan",
    "🚗 Car Loan",
    "📱 Appliance & Gadget Loan",
    "📊 Overall Summary"
])


# =========================================================
# 🏠 HOME LOAN ADVISOR
# =========================================================
with tab_home:
    st.header("🏠 Home Loan Advisor")

    h1, h2, h3, h4 = st.columns(4)
    home_property = h1.number_input("Property Value (₹)", min_value=0, value=3000000,
                                    step=50000, key="home_property")
    home_loan = h2.number_input("Loan Amount Required (₹)", min_value=10000, value=2500000,
                                step=10000, key="home_loan")
    home_rate = h3.number_input("Annual Interest Rate (%)", min_value=1.0, max_value=20.0,
                                value=8.5, step=0.1, key="home_rate")
    home_years = h4.slider("Loan Tenure (years)", min_value=1, max_value=30, value=20,
                           key="home_years")

    home_emi = calculate_emi(home_loan, home_rate, home_years)
    home_total_payment = home_emi * home_years * 12
    home_total_interest = home_total_payment - home_loan
    home_checks = run_home_checks(monthly_income, existing_emis, home_loan, home_rate,
                                  home_years, age, credit_score, home_property)

    ht1, ht2, ht3, ht4, ht5, ht6, ht7 = st.tabs([
        "📊 EMI Calculator", "✅ Eligibility Check", "📅 Repayment Schedule",
        "💸 Prepayment Simulator", "🏦 Compare Banks", "🧾 Tax Benefits", "🤖 AI Advice"
    ])

    with ht1:
        st.subheader("EMI Calculator")
        show_emi_summary(home_loan, home_emi, home_total_interest, home_total_payment)

        report = create_summary_report(
            "HOME LOAN - SUMMARY REPORT",
            [("Age", age), ("Monthly Income", format_inr(monthly_income)),
             ("Existing EMIs", format_inr(existing_emis)), ("Credit Score", credit_score),
             ("Property Value", format_inr(home_property)), ("Loan Amount", format_inr(home_loan)),
             ("Interest Rate", f"{home_rate}%"), ("Tenure", f"{home_years} years")],
            [("Monthly EMI", format_inr(home_emi)),
             ("Total Interest", format_inr(home_total_interest)),
             ("Total Payment", format_inr(home_total_payment))],
            home_checks)
        st.download_button("⬇️ Download Summary Report (.txt)", report,
                           "home_loan_summary.txt", "text/plain", key="home_report")

    with ht2:
        st.subheader("Home Loan Eligibility Check")
        st.caption("Based on the FOIR rule (total EMIs should not exceed 50% of your income), "
                   "the property's Loan-to-Value limit, your credit score and your age.")
        show_eligibility_result(home_checks, existing_emis, home_emi, monthly_income)

    with ht3:
        show_schedule_tab(home_loan, home_rate, home_years * 12, "home",
                          "home_loan_schedule.csv", show_yearly=True)

    with ht4:
        show_prepayment_tab(home_loan, home_rate, home_years, "home")

    with ht5:
        show_bank_comparison_tab(home_loan, home_rate, home_years, [8.35, 8.75, 9.10], "home")

    with ht6:
        st.subheader("Income Tax Benefits on Your Home Loan")
        st.caption("Estimate for a self-occupied house under the OLD tax regime: "
                   f"principal repaid up to {format_inr(TAX_LIMIT_80C)} per year (Section 80C) and "
                   f"interest paid up to {format_inr(TAX_LIMIT_24B)} per year (Section 24(b)).")

        tax_slab = st.selectbox("Your income tax slab (%)", [5, 20, 30], index=1,
                                key="home_tax_slab")
        home_yearly = generate_yearly_summary(
            generate_amortization_schedule(home_loan, home_rate, home_years))
        tax_df = calculate_tax_benefit(home_yearly, tax_slab)

        t1, t2 = st.columns(2)
        t1.metric("Tax Saved in Year 1", format_inr(tax_df.loc[0, "Tax Saved"]))
        t2.metric("Total Tax Saved Over the Loan", format_inr(tax_df["Tax Saved"].sum()))
        st.dataframe(tax_df)
        st.caption("⚠️ Simplified estimate. The 80C limit is shared with your other investments "
                   "(PPF, EPF, life insurance, etc.), cess is ignored, and these deductions are "
                   "not available under the new tax regime. Tax rules change, so please "
                   "verify with a tax professional.")

    with ht7:
        home_profile = f"""
        Loan type: Home loan
        Age: {age}
        Monthly Income: ₹{monthly_income}
        Existing Monthly EMIs: ₹{existing_emis}
        Credit Score: {credit_score}
        Property Value: ₹{home_property}
        Requested Loan Amount: ₹{home_loan}
        Interest Rate: {home_rate}%
        Tenure: {home_years} years
        Calculated EMI: ₹{home_emi:,.2f}
        Eligible: {home_checks['overall']}
        Max Eligible Loan: ₹{home_checks['foir']['max_eligible_loan']:,.2f}
        Eligibility checks:
        {checks_to_text(home_checks['checks'])}
        """
        show_ai_tab("home loan", home_profile, "home")


# =========================================================
# 🚗 CAR LOAN ADVISOR
# =========================================================
with tab_car:
    st.header("🚗 Car Loan Advisor")

    c1, c2, c3, c4 = st.columns(4)
    car_price = c1.number_input("Car On-Road Price (₹)", min_value=100000, value=800000,
                                step=10000, key="car_price")
    car_down_pct = c2.slider("Down Payment (%)", min_value=0, max_value=90, value=20,
                             step=5, key="car_down_pct")
    car_rate = c3.number_input("Annual Interest Rate (%)", min_value=1.0, max_value=25.0,
                               value=9.0, step=0.1, key="car_rate")
    car_years = c4.slider("Loan Tenure (years)", min_value=1, max_value=7, value=5,
                          key="car_years")

    c5, c6, c7 = st.columns(3)
    car_fuel = c5.number_input("Monthly Fuel + Maintenance (₹)", min_value=0, value=5000,
                               step=500, key="car_fuel")
    car_insurance = c6.number_input("Yearly Insurance (₹)", min_value=0, value=25000,
                                    step=1000, key="car_insurance")
    car_depreciation = c7.slider("Assumed Yearly Drop in Car Value (%)", min_value=5,
                                 max_value=30, value=15, key="car_depreciation")

    car_down_payment = car_price * car_down_pct / 100
    car_loan = car_price - car_down_payment
    car_months = car_years * 12
    car_emi = calculate_emi(car_loan, car_rate, car_years)
    car_total_payment = car_emi * car_months
    car_total_interest = car_total_payment - car_loan
    car_running_cost = car_fuel + car_insurance / 12    # per month
    car_checks = run_car_checks(monthly_income, existing_emis, car_loan, car_price,
                                car_rate, car_years, age, credit_score)
    car_smart = car_smart_rule_checks(car_price, car_down_payment, car_years, car_emi,
                                      car_running_cost, monthly_income)

    st.caption(f"Down payment: {format_inr(car_down_payment)}  |  "
               f"Loan amount: {format_inr(car_loan)}")

    ct1, ct2, ct3, ct4, ct5, ct6, ct7 = st.tabs([
        "📊 EMI & True Cost", "✅ Eligibility & Smart Rule", "📅 Repayment Schedule",
        "📉 Loan vs Car Value", "💸 Prepayment", "🏦 Compare Banks", "🤖 AI Advice"
    ])

    with ct1:
        st.subheader("Car Loan EMI")
        show_emi_summary(car_loan, car_emi, car_total_interest, car_total_payment)

        st.markdown("---")
        st.subheader("True Cost of Owning the Car")
        st.write("The EMI is not the only cost. Fuel, maintenance and insurance add up too.")

        true_monthly = car_emi + car_running_cost
        total_ownership = car_down_payment + car_total_payment + car_running_cost * car_months
        if monthly_income > 0:
            true_pct = true_monthly / monthly_income * 100
        else:
            true_pct = 100

        o1, o2, o3 = st.columns(3)
        o1.metric("True Monthly Cost (EMI + running)", format_inr(true_monthly))
        o2.metric("Share of Your Monthly Income", f"{true_pct:.1f}%")
        o3.metric(f"Total Cost Over {car_years} Years", format_inr(total_ownership))

        report = create_summary_report(
            "CAR LOAN - SUMMARY REPORT",
            [("Age", age), ("Monthly Income", format_inr(monthly_income)),
             ("Existing EMIs", format_inr(existing_emis)), ("Credit Score", credit_score),
             ("Car Price", format_inr(car_price)), ("Down Payment", format_inr(car_down_payment)),
             ("Loan Amount", format_inr(car_loan)), ("Interest Rate", f"{car_rate}%"),
             ("Tenure", f"{car_years} years")],
            [("Monthly EMI", format_inr(car_emi)),
             ("Total Interest", format_inr(car_total_interest)),
             ("True Monthly Cost", format_inr(true_monthly)),
             (f"Total Cost ({car_years} yrs)", format_inr(total_ownership))],
            car_checks)
        st.download_button("⬇️ Download Summary Report (.txt)", report,
                           "car_loan_summary.txt", "text/plain", key="car_report")

    with ct2:
        st.subheader("Car Loan Eligibility Check")
        show_eligibility_result(car_checks, existing_emis, car_emi, monthly_income)

        st.markdown("---")
        st.subheader("Smart-Buying Check: the 20/4/10 Rule")
        st.caption("A popular guideline for buying a car sensibly: pay at least 20% down, "
                   "take the loan for at most 4 years, and keep the total monthly car cost "
                   "(EMI + fuel + insurance + maintenance) within 10% of your income. "
                   "These are tips, not bank rules, so they only show warnings.")
        show_checks(car_smart)

    with ct3:
        show_schedule_tab(car_loan, car_rate, car_months, "car",
                          "car_loan_schedule.csv", show_yearly=True)

    with ct4:
        st.subheader("Loan Balance vs Car Value")
        st.write("A car loses value every year. If you still owe more than the car is worth, "
                 "you are 'underwater' - selling the car would not even clear the loan.")

        car_schedule = generate_schedule_months(car_loan, car_rate, car_months)
        car_values = estimate_car_values(car_price, car_months, car_depreciation)
        balances = list(car_schedule["Remaining Balance"])
        months_underwater = sum(1 for b, v in zip(balances, car_values) if b > v)

        v1, v2, v3 = st.columns(3)
        v1.metric("Car Value at End of Loan", format_inr(car_values[-1]))
        v2.metric("Months You Owe More Than Car Is Worth", months_underwater)
        v3.metric("Total Value Lost", format_inr(car_price - car_values[-1]))

        if months_underwater > 0:
            st.warning(f"⚠️ For about {format_tenure(months_underwater)} you may owe more than the "
                       "car is worth. A bigger down payment or a shorter tenure reduces this risk.")
        else:
            st.success("✅ Your loan balance stays below the estimated car value the whole time.")

        fig, ax = plt.subplots()
        ax.plot(range(1, car_months + 1), balances, label="Loan balance")
        ax.plot(range(1, car_months + 1), car_values, label="Estimated car value")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (₹)")
        ax.set_title("Loan Balance vs Estimated Car Value")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Assumes the car loses {car_depreciation}% of its value every year. "
                   "Real resale values depend on the model, condition and market.")

    with ct5:
        show_prepayment_tab(car_loan, car_rate, car_years, "car")

    with ct6:
        show_bank_comparison_tab(car_loan, car_rate, car_years, [9.0, 9.5, 10.25], "car")

    with ct7:
        car_profile = f"""
        Loan type: Car loan
        Age: {age}
        Monthly Income: ₹{monthly_income}
        Existing Monthly EMIs: ₹{existing_emis}
        Credit Score: {credit_score}
        Car Price: ₹{car_price}
        Down Payment: ₹{car_down_payment:,.0f} ({car_down_pct}%)
        Loan Amount: ₹{car_loan:,.0f}
        Interest Rate: {car_rate}%
        Tenure: {car_years} years
        Calculated EMI: ₹{car_emi:,.2f}
        Monthly fuel + maintenance + insurance: ₹{car_running_cost:,.0f}
        Eligible: {car_checks['overall']}
        Eligibility checks:
        {checks_to_text(car_checks['checks'])}
        20/4/10 rule checks:
        {checks_to_text(car_smart)}
        """
        show_ai_tab("car loan", car_profile, "car")


# =========================================================
# 📱 APPLIANCE & GADGET LOAN ADVISOR
# =========================================================
with tab_gadget:
    st.header("📱 Appliance & Gadget Loan Advisor")
    st.caption("For phones, laptops, TVs, fridges, ACs and other consumer durables. "
               "These loans are short (months, not years) and usually cost more per year "
               "than home or car loans.")

    g1, g2, g3 = st.columns(3)
    gadget_product = g1.selectbox("Product Type", list(PRODUCT_LIFE_YEARS.keys()),
                                  key="gadget_product")
    gadget_price = g2.number_input("Product Price (₹)", min_value=1000, value=60000,
                                   step=1000, key="gadget_price")
    gadget_down_pct = g3.slider("Down Payment (%)", min_value=0, max_value=90, value=10,
                                step=5, key="gadget_down_pct")

    g4, g5, g6 = st.columns(3)
    gadget_rate = g4.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=36.0,
                                  value=15.0, step=0.5, key="gadget_rate")
    gadget_months = g5.slider("Loan Tenure (months)", min_value=3, max_value=36, value=12,
                              key="gadget_months")
    gadget_fee_pct = g6.number_input("Processing Fee (% of loan)", min_value=0.0, max_value=10.0,
                                     value=1.0, step=0.25, key="gadget_fee_pct")

    gadget_down = gadget_price * gadget_down_pct / 100
    gadget_loan = gadget_price - gadget_down
    gadget_emi = calculate_emi_months(gadget_loan, gadget_rate, gadget_months)
    gadget_total_emis = gadget_emi * gadget_months
    gadget_interest = gadget_total_emis - gadget_loan
    gadget_fee = gadget_loan * gadget_fee_pct / 100
    gadget_total_cost = gadget_down + gadget_total_emis + gadget_fee
    gadget_extra_cost = gadget_total_cost - gadget_price     # interest + fee
    gadget_checks = run_gadget_checks(monthly_income, existing_emis, gadget_loan,
                                      gadget_rate, gadget_months, age, credit_score)
    gadget_smart = gadget_smart_checks(gadget_product, gadget_price, gadget_emi,
                                       gadget_months, gadget_extra_cost, monthly_income)

    st.caption(f"Down payment: {format_inr(gadget_down)}  |  "
               f"Loan amount: {format_inr(gadget_loan)}")

    gt1, gt2, gt3, gt4, gt5, gt6 = st.tabs([
        "📊 EMI & Total Cost", "✅ Eligibility & Smart Checks", "🏷️ No-Cost EMI Checker",
        "⏱️ Tenure Comparison", "📅 Repayment Schedule", "🤖 AI Advice"
    ])

    with gt1:
        st.subheader("Gadget Loan EMI & Total Cost")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Monthly EMI", format_inr(gadget_emi))
        k2.metric("Total Interest", format_inr(gadget_interest))
        k3.metric("Processing Fee", format_inr(gadget_fee))
        k4.metric("Extra Cost Over Price", format_inr(gadget_extra_cost))
        st.write(f"**Total you will pay:** {format_inr(gadget_total_cost)} "
                 f"for a {gadget_product.lower()} priced at {format_inr(gadget_price)}.")

        show_pie_chart(["Product Price", "Interest", "Processing Fee"],
                       [gadget_price, gadget_interest, gadget_fee],
                       "Where Your Money Goes")

        # "Save up instead" insight
        if gadget_emi > 0:
            months_to_save = math.ceil(gadget_loan / gadget_emi)
            st.info(f"💡 If you saved {format_inr(gadget_emi)} every month instead of borrowing, "
                    f"you would have {format_inr(gadget_loan)} in about {months_to_save} months "
                    f"and pay no interest. Buying on EMI gets you the product now, but it "
                    f"costs {format_inr(gadget_extra_cost)} extra.")

        report = create_summary_report(
            "GADGET / APPLIANCE LOAN - SUMMARY REPORT",
            [("Age", age), ("Monthly Income", format_inr(monthly_income)),
             ("Existing EMIs", format_inr(existing_emis)), ("Credit Score", credit_score),
             ("Product", gadget_product), ("Product Price", format_inr(gadget_price)),
             ("Down Payment", format_inr(gadget_down)), ("Loan Amount", format_inr(gadget_loan)),
             ("Interest Rate", f"{gadget_rate}%"), ("Tenure", f"{gadget_months} months"),
             ("Processing Fee", format_inr(gadget_fee))],
            [("Monthly EMI", format_inr(gadget_emi)),
             ("Total Interest", format_inr(gadget_interest)),
             ("Total You Pay", format_inr(gadget_total_cost)),
             ("Extra Over Price", format_inr(gadget_extra_cost))],
            gadget_checks)
        st.download_button("⬇️ Download Summary Report (.txt)", report,
                           "gadget_loan_summary.txt", "text/plain", key="gadget_report")

    with gt2:
        st.subheader("Gadget Loan Eligibility Check")
        show_eligibility_result(gadget_checks, existing_emis, gadget_emi, monthly_income)

        st.markdown("---")
        st.subheader("Smart-Buying Checks")
        st.caption("Tips to avoid overpaying for a gadget. These are advice, not lender rules, "
                   "so they only show warnings.")
        show_checks(gadget_smart)

    with gt3:
        st.subheader("Is This 'No-Cost EMI' Really Free?")
        st.write("Shops often advertise 'No-Cost EMI'. But the interest is usually hidden: "
                 "either the EMI price is higher than the pay-in-full price, or you lose a "
                 "cash discount. Enter the offer below to find the hidden cost.")

        n1, n2, n3, n4 = st.columns(4)
        nc_cash_price = n1.number_input("Price if you pay in full today (₹)", min_value=1000,
                                        value=57000, step=500, key="nc_cash_price")
        nc_emi = n2.number_input("Monthly EMI in the offer (₹)", min_value=100,
                                 value=10000, step=100, key="nc_emi")
        nc_months = n3.slider("EMI months", min_value=3, max_value=36, value=6, key="nc_months")
        nc_fee = n4.number_input("Upfront / processing fee (₹)", min_value=0,
                                 value=0, step=100, key="nc_fee")

        nc_total_paid = nc_emi * nc_months + nc_fee
        nc_hidden_cost = nc_total_paid - nc_cash_price
        nc_effective_rate = calculate_effective_rate(nc_cash_price, nc_emi, nc_months, nc_fee)

        r1, r2, r3 = st.columns(3)
        r1.metric("Total Paid Through EMIs", format_inr(nc_total_paid))
        r2.metric("Hidden Cost vs Paying in Full", format_inr(nc_hidden_cost))
        r3.metric("Real Interest Rate (per year)", f"{nc_effective_rate:.1f}%")

        if nc_hidden_cost <= 0:
            st.success("✅ This offer is truly no-cost: you pay no more than the pay-in-full price.")
        else:
            st.warning(f"⚠️ It is not free. You pay {format_inr(nc_hidden_cost)} extra, which is "
                       f"like borrowing at about {nc_effective_rate:.1f}% per year.")
            if nc_effective_rate < gadget_rate:
                st.write(f"That is lower than the {gadget_rate}% loan rate you entered above, "
                         "so this offer may still be the cheaper option.")
            else:
                st.write(f"That is not lower than the {gadget_rate}% loan rate you entered "
                         "above, so compare other options before choosing.")

    with gt4:
        st.subheader("Tenure Comparison")
        st.write("A longer tenure means a smaller EMI but more total interest. "
                 f"Here is your {format_inr(gadget_loan)} loan at {gadget_rate}% for different tenures:")

        tenure_df = tenure_comparison(gadget_loan, gadget_rate, [3, 6, 9, 12, 18, 24, 36])
        st.dataframe(tenure_df)

        chart1, chart2 = st.columns(2)
        with chart1:
            st.caption("Monthly EMI by tenure")
            st.bar_chart(tenure_df.set_index("Tenure (months)")["Monthly EMI (₹)"])
        with chart2:
            st.caption("Total interest by tenure")
            st.bar_chart(tenure_df.set_index("Tenure (months)")["Total Interest (₹)"])

        shortest = tenure_df.iloc[0]
        longest = tenure_df.iloc[-1]
        st.write(f"Paying over {shortest['Tenure (months)']} months costs "
                 f"{format_inr(longest['Total Interest (₹)'] - shortest['Total Interest (₹)'])} "
                 f"less interest than paying over {longest['Tenure (months)']} months, "
                 "but the EMI is much higher.")

    with gt5:
        show_schedule_tab(gadget_loan, gadget_rate, gadget_months, "gadget",
                          "gadget_loan_schedule.csv", show_yearly=False)

    with gt6:
        gadget_profile = f"""
        Loan type: Consumer durable / gadget loan for a {gadget_product}
        Age: {age}
        Monthly Income: ₹{monthly_income}
        Existing Monthly EMIs: ₹{existing_emis}
        Credit Score: {credit_score}
        Product Price: ₹{gadget_price}
        Down Payment: ₹{gadget_down:,.0f} ({gadget_down_pct}%)
        Loan Amount: ₹{gadget_loan:,.0f}
        Interest Rate: {gadget_rate}%
        Tenure: {gadget_months} months
        Processing Fee: ₹{gadget_fee:,.0f}
        Calculated EMI: ₹{gadget_emi:,.2f}
        Extra cost over the price (interest + fee): ₹{gadget_extra_cost:,.0f}
        Eligible: {gadget_checks['overall']}
        Eligibility checks:
        {checks_to_text(gadget_checks['checks'])}
        Smart-buying checks:
        {checks_to_text(gadget_smart)}
        """
        show_ai_tab("gadget loan", gadget_profile, "gadget")


# =========================================================
# 📊 OVERALL SUMMARY (all loans together)
# =========================================================
with tab_summary:
    st.header("📊 Overall Summary")
    st.write("What would your monthly EMIs look like if you took ALL the loans "
             "entered in the other tabs? (Only counts loans you actually plan to take.)")

    emi_table = pd.DataFrame({
        "Loan": ["🏠 Home Loan", "🚗 Car Loan", "📱 Gadget Loan"],
        "Loan Amount (₹)": [round(home_loan), round(car_loan), round(gadget_loan)],
        "Monthly EMI (₹)": [round(home_emi), round(car_emi), round(gadget_emi)],
        "Tenure": [f"{home_years} years", f"{car_years} years", f"{gadget_months} months"],
    })
    st.dataframe(emi_table)

    new_emis = home_emi + car_emi + gadget_emi
    all_emis = existing_emis + new_emis

    s1, s2, s3 = st.columns(3)
    s1.metric("Existing EMIs", format_inr(existing_emis))
    s2.metric("New EMIs (all three loans)", format_inr(new_emis))
    s3.metric("Total Monthly EMIs", format_inr(all_emis))

    if monthly_income > 0:
        total_ratio = all_emis / monthly_income * 100
    else:
        total_ratio = 100
    st.write(f"**Total EMIs as a share of your income: {total_ratio:.1f}%**")
    st.progress(min(total_ratio / 100, 1.0))

    if total_ratio > 50:
        st.error("❌ Together these loans would be too heavy. Lenders usually stop at about 50% "
                 "of income, so you would probably not get approved for all of them.")
    elif total_ratio > 40:
        st.warning("⚠️ This is a heavy EMI load. Try to keep total EMIs under 40% of your income "
                   "so you still have room for savings and emergencies.")
    else:
        st.success("✅ Your total EMIs would stay in a comfortable range.")

    st.bar_chart(emi_table.set_index("Loan")["Monthly EMI (₹)"])

st.markdown("---")
st.caption("Project RE-03 | Loan Advisor (Home, Car, Gadget) | Built with Streamlit + Google Gemini API")
