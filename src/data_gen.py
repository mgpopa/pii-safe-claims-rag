import argparse, os, pandas as pd, random
from faker import Faker
from tqdm import trange

LOB_CHOICES = ["Auto", "Property", "Liability", "Travel", "Health"]
CAUSES = ["rear-ended", "hail damage", "burst pipe", "slip and fall", "stolen laptop", "windshield crack"]
CITIES = ["Zurich", "Geneva", "Basel", "Lausanne", "Bern", "Lugano", "St. Gallen"]

PHONE_STYLES = ["ch_strict"] # swiss style, mix

def synth_phone(style="ch_strict", rng=None):
    rng = rng or random
    cc = "+41" if style == "ch_strict" else rng.choice(["+41", "+43", "+49", "+44", "+39"])
    area = rng.choice(["21","22","31","32","33","41","61","71","81","79", "78", "76", "44", "43"])
    local = f"{rng.randint(100,999)} {rng.randint(10,99)} {rng.randint(10,99)}"

    if style == "ch_strict":
        # +41 79 123 45 67 or 044 123 45 67
        if rng.random() < 0.6:
            return f"{cc} {area} {local}"
    else:
        return f"0{area} {local}"
    
    #multi region/messy
    us_area = rng.randint(201,998)
    us_mid = rng.randint(100,999)
    us_last = rng.randint(1000,9999)
    ext = f" x.{rng.randint(100,9999)}" if rng.random() < 0.25 else ""

    choice = rng.random()
    if choice < 0.2:
        return f"{cc} {area} {local.replace(' ', ' ')}{ext}"
    elif choice < 0.4:
        return f"{cc} ({area}) {local}{ext}"
    elif choice < 0.55:
        return f"({us_area}) {us_mid}-{us_last}{ext}"
    elif choice < 0.7:
        return f"{us_area}-{us_mid}-{us_last}{ext}"
    elif choice < 0.85:
        return f"{us_area}.{us_mid}.{us_last}{ext}"
    else:
        return f"+1 {us_area} {us_mid} {us_last}{ext}"

SWISS_MOBILE = ["075","076","077","078","079"]
SWISS_AREA = ["021","022","024","026","027","031","032","033","034","041","043","044","061","062","071","081","091"]
def synth_phone_ch_strict(rng=None):
    rng = rng or random
    if rng.random() < 0.6:
        pfx = rng.choice(SWISS_MOBILE)
    else:
        pfx = rng.choice(SWISS_AREA)
    
    part1 = rng.randint(100,999)
    part2 = rng.randint(10,99)
    part3 = rng.randint(10,99)

    if rng.random() < 0.6:
        return f"+41 {pfx[1:]} {part1} {part2} {part3}" # +41 79 123 45 67 / +41 44 123 45 67
    else:
        return f"{pfx} {part1} {part2} {part3}"          # 079 123 45 67 / 044 123 45 67   

def synth_policy_id(faker):
    left = faker.bothify(text="??#").upper()
    mid = faker.bothify(text="###").upper()
    end = faker.bothify(text="##").upper()
    return f"{left}-{mid}-{end}"

def generate(n, out_csv, phone_style="ch_strict"):
    faker = Faker()
    rows = []
    for i in trange(n):
        claim_id = f"C{i:06d}"
        lob = random.choice(LOB_CHOICES)
        city = random.choice(CITIES)
        cause = random.choice(CAUSES)
        name = faker.name()
        email = faker.email()
        phone = synth_phone_ch_strict()
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data/claims.csv")
    ap.add_argument("--phone_style", choices=PHONE_STYLES, default="ch_strict",
                    help="Phone number format profile: ch_strict (default) or multi")
    args = ap.parse_args()
    generate(args.n, args.out, phone_style=args.phone_style)
