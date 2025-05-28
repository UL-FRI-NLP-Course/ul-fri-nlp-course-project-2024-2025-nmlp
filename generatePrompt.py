import pandas as pd

# Load Excel
df = pd.read_excel("data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx")

# Convert 'Datum' to datetime if necessary
df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')

# Filter: only events from one day (e.g., 30 June 2024)
date_filter = pd.Timestamp("2024-06-30")
filtered_df = df[df['Datum'] == date_filter]

# Now extract important content columns
important_contents = []

for idx, row in filtered_df.iterrows():
    for column in [
        "ContentPomembnoSLO",
        "ContentNesreceSLO",
        "ContentZastojiSLO",
        "ContentOvireSLO",
        "ContentDeloNaCestiSLO",
        "ContentOpozorilaSLO",
    ]:
        content = row.get(column, "")
        if isinstance(content, str) and content.strip():  # Not empty
            important_contents.append(content.strip())

# Build the prompt
prompt_data = ""
for event in important_contents:
    prompt_data += f"- {event}\n"

final_prompt = f"""
Iz danih podatkov o prometu ustvarite prometna poročila v slovenščini, v stilu Radia Slovenija.

Uporabite naslednja pravila:
- Pravilna imena cest in smeri.
- Formulacija: cesta in smer + razlog + posledica in odsek ALI razlog + cesta in smer + posledica in odsek.
- Poročajte samo o pomembnih dogodkih (nesreče, zaprte ceste, dela itd.).
- Vsako poročilo naj bo kratko (2–3 stavki).
- Jasno ločite več dogodkov.

Podatki:
{prompt_data}

Generiraj poročila:
"""

with open("prompt_for_gams.txt", "w", encoding="utf-8") as f:
    f.write(final_prompt)

print("Prompt saved!")
