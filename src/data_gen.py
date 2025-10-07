import argparse, os, pandas as pd, random
from faker import Faker
from tqdm import trange

LOB_CHOICES = ["Auto", "Property", "Liability", "Travel", "Health"]
CAUSES = ["rear-ended", "hail damage", "burst pipe", "slip and fall", "stolen laptop", "windshield crack"]
CITIES = ["Zurich", "Geneva", "Basel", "Lausanne", "Bern", "Lugano", "St. Gallen"]

def synth_policy_id(faker):
    left = faker.bothify(text="??#").upper()
    mid = faker.bothify(text="###").upper()
    end = faker.bothify(text="##").upper()
    return f"{left}-{mid}-{end}"

def generate(n, out_csv):
    faker = Faker()
    rows = []
    for i in trange(n):
        claim_id = f"C{i:06d}"
        lob = random.choice(LOB_CHOICES)
        city = random.choice(CITIES)
        cause = random.choice(CAUSES)
        name = faker.name()
        email = faker.email()
        phone = faker.phone_number()
        policy_id = synth_policy_id(faker)
        amount = round(random.uniform(200, 15000), 2)
        desc = f"{cause} near {city}. Minor injuries reported."
        note = (
            f"Spoke with {name} ({email}, {phone}) regarding claim {claim_id}. "
            f"Policy {policy_id}. {lob} loss in {city}. "
            f"Vehicle towed; estimate CHF {amount}. "
            f"Insured states they were {cause} at traffic light."
        )
        rows.append({
            "claim_id": claim_id,
            "lob": lob,
            "city": city,
            "policy_id": policy_id,
            "contact_name": name,
            "contact_email": email,
            "contact_phone": phone,
            "amount": amount,
            "loss_description": desc,
            "adjuster_note": note
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows.")
