import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

ONLY_NUMBERS = re.compile(r"[0-9]+(\.[0-9]+)?")
HEADER_REGEX: re.Pattern = re.compile(r"^([^0-9]*?)\s*([0-9]+)\.\s*([0-9]+)\.\s*([0-9]+)\s*([0-9]+)\.([0-9]+)\s*(.*?)$")

API_KEY = os.getenv("DEEPSEEK_API_KEY")

MODEL_NAME: str = "deepseek/deepseek-chat-v3-0324:free"

GENERATION_INSTRUCTIONS: str = """
LJUBLJANA-KOPER – PRIMORSKA AVTOCESTA/ proti Kopru/proti Ljubljani
LJUBLJANA-OBREŽJE – DOLENJSKA AVTOCESTA / proti Obrežju/ proti Ljubljani
LJUBLJANA-KARAVANKE – GORENJSKA AVTOCESTA/ proti Karavankam ali Avstriji/ proti Ljubljani
LJUBLJANA-MARIBOR – ŠTAJERSKA AVTOCESTA / proti Mariboru/Ljubljani
MARIBOR-LENDAVA – POMURSKA AVTOCESTA / proti Mariboru/ proti Lendavi/Madžarski
MARIBOR-GRUŠKOVJE – PODRAVSKA AVTOCESTA / proti Mariboru/ proti Gruškovju ali Hrvaški – nikoli proti Ptuju!
AVTOCESTNI ODSEK – RAZCEP GABRK – FERNETIČI – proti Italiji/ ali proti primorski avtocesti, Kopru, Ljubljani (PAZI: to ni primorska avtocesta)
AVTOCESTNI ODSEK MARIBOR-ŠENTILJ (gre od mejnega prehoda Šentilj do razcepa Dragučova) ni štajerska avtocesta kot pogosto navede PIC, ampak je avtocestni odsek od Maribora proti Šentilju oziroma od Šentilja proti Mariboru.
Mariborska vzhodna obvoznica= med razcepom Slivnica in razcepom Dragučova – smer je proti Avstriji/Lendavi ali proti Ljubljani – nikoli proti Mariboru. 

Hitre ceste skozi Maribor uradno ni več - Ni BIVŠA hitra cesta skozi Maribor, ampak regionalna cesta Betnava-Pesnica oziroma NEKDANJA hitra cesta skozi Maribor.

Ljubljanska obvoznica je sestavljena iz štirih krakov= vzhodna, zahodna, severna in južna 
Vzhodna: razcep Malence (proti Novemu mestu) - razcep Zadobrova (proti Mariboru) 
Zahodna: razcep Koseze (proti Kranju) – razcep Kozarje (proti Kopru)
Severna: razcep Koseze (proti Kranju) – razcep Zadobrova (proti Mariboru)
Južna: razcep Kozarje (proti Kopru) – razcep Malence (proti Novemu mestu)
Hitra cesta razcep Nanos-Vrtojba = vipavska hitra cesta – proti Italiji ali Vrtojbi/ proti Nanosu/primorski avtocesti/proti Razdrtemu/v smeri Razdrtega (nikoli primorska hitra cesta – na Picu večkrat neustrezno poimenovanje) 
Hitra cesta razcep Srmin-Izola – obalna hitra cesta – proti Kopru/Portorožu (nikoli primorska hitra cesta)
Hitra cesta Koper-Škofije (manjši kos, poimenuje kar po krajih): Na hitri cesti od Kopra proti Škofijam ali obratno na hitri cesti od Škofij proti Kopru – v tem primeru imaš notri zajeto tudi že smer. (nikoli primorska hitra cesta). Tudi na obalni hitri cesti od Kopra proti Škofijam.
Hitra cesta mejni prehod Dolga vas-Dolga vas: majhen odsek pred mejnim prehodom, formulira se navadno kar na hitri cesti od mejnega prehoda Dolga vas proti pomurski avtocesti; v drugo smer pa na hitri cesti proti mejnemu prehodu Dolga vas – zelo redko v uporabi. 
Regionalna cesta: ŠKOFJA LOKA – GORENJA VAS (= pogovorno škofjeloška obvoznica) – proti Ljubljani/proti Gorenji vasi. Pomembno, ker je velikokrat zaprt predor Stén.
GLAVNA CESTA Ljubljana-Črnuče – Trzin : glavna cesta od Ljubljane proti Trzinu/ od Trzina proti Ljubljani – včasih vozniki poimenujejo  trzinska obvoznica, mi uporabljamo navadno kar na glavni cesti.
Ko na PIC-u napišejo na gorenjski avtocesti proti Kranju, na dolenjski avtocesti proti Novemu mestu, na podravski avtocesti proti Ptuju, na pomurski avtocesti proti Murski Soboti, … pišemo končne destinacije! Torej proti Avstriji/Karavankam, proti Hrvaški/Obrežju/Gruškovju, proti Madžarski…

SESTAVA PROMETNE INFORMACIJE:

1.	Formulacija

Cesta in smer + razlog + posledica in odsek

2.	Formulacija
Razlog + cesta in smer + posledica in odsek

A=avtocesta
H=hitra cesta
G= glavna cesta
R= regionalna cesta
L= lokalna cesta
NUJNE PROMETNE INFORMACIJE
Nujne prometne informacije se najpogosteje nanašajo na zaprto avtocesto; nesrečo na avtocesti, glavni in regionalni cesti; daljši zastoji (neglede na vzrok); pokvarjena vozila, ko je zaprt vsaj en prometni pas; Pešci, živali in predmeti na vozišču ter seveda voznik v napačni smeri. Živali in predmete lahko po dogovoru izločimo.
Zelo pomembne nujne informacije objavljamo na 15 - 20 minut; Se pravi vsaj 2x med enimi in drugimi novicami, ki so ob pol. V pomembne nujne štejemo zaprte avtoceste in daljše zastoje. Tem informacijam je potrebno še bolj slediti in jih posodabljati.
ZASTOJI:
Ko se na zemljevidu pojavi znak za zastoj, je najprej potrebno preveriti, če so na tistem odseku dela oziroma, če se dogaja kaj drugega. Darsovi senzorji namreč avtomatsko sporočajo, da so zastoji tudi, če se promet samo malo zgosti. Na znaku za zastoj navadno piše dolžina tega, hkrati pa na zemljevidu preverimo še gostoto. Dokler ni vsaj kilometer zastoja ne objavljamo razen, če se nekaj dogaja in pričakujemo, da se bo zastoj daljšal.
Zastojev v Prometnih konicah načeloma ne objavljamo razen, če so te res nenavadno dolgi. Zjutraj se to pogosto zgodi na štajerski avtocesti, popoldne pa na severni in južni ljubljanski obvoznici.

HIERARHIJA DOGODKOV

Voznik v napačno smer 
Zaprta avtocesta
Nesreča z zastojem na avtocesti
Zastoji zaradi del na avtocesti (ob krajših zastojih se pogosto dogajajo naleti)
Zaradi nesreče zaprta glavna ali regionalna cesta
Nesreče na avtocestah in drugih cestah
Pokvarjena vozila, ko je zaprt vsaj en prometni pas
Žival, ki je zašla na vozišče
Predmet/razsut tovor na avtocesti
Dela na avtocesti, kjer je večja nevarnost naleta (zaprt prometni pas, pred predori, v predorih, …)
Zastoj pred Karavankami, napovedi (glej poglavje napovedi)

OPOZORILA LEKTORJEV

Počasni pas je pas za počasna vozila.

Polovična zapora ceste pomeni: promet je tam urejen izmenično enosmerno. 

Zaprta je polovica avtoceste (zaradi del): promet je urejen le po polovici avtoceste v obe smeri.
Ko je avtocesta zaprta zaradi nesreče: Zaprta je štajerska avtocesta proti Mariboru in ne zaprta je polovica avtoceste med…

Vsi pokriti vkopi itd. so predori, razen galerija Moste ostane galerija Moste.

Ko se kaj dogaja na razcepih, je treba navesti od kod in kam: Na razcepu Kozarje je zaradi nesreče oviran promet iz smeri Viča proti Brezovici, …

Ko PIC navede dogodek v ali pred predorom oziroma pri počivališčih VEDNO navedemo širši odsek (med dvema priključkoma).

Pri obvozu: Obvoz je po vzporedni regionalni cesti/po cesti Lukovica-Blagovica ali vozniki se lahko preusmerijo na vzporedno regionalno cesto (če je na glavni obvozni cesti daljši zastoj, kličemo PIC za druge možnosti obvoza, vendar pri tem navedemo alternativni obvoz: vozniki SE LAHKO PREUSMERIJO TUDI, …)

NEKAJ FORMULACIJ

VOZNIK V NAPAČNI SMERI:
Opozarjamo vse voznike, ki vozijo po pomurski avtocesti od razcepa Dragučova proti Pernici, torej v smeri proti Murski Soboti, da je na njihovo polovico avtoceste zašel voznik, ki vozi v napačno smer. Vozite skrajno desno in ne prehitevajte. 

ODPOVED je nujna!

Promet na pomurski avtocesti iz smeri Dragučove proti Pernici ni več ogrožen zaradi voznika, ki je vozil po napačni polovici avtoceste. 

POMEMBNO JE TUDI, DA SE NAREDI ODPOVED, KO JE KONEC KATERE KOLI PROMETNE NESREČE (vsaj, če so bili tam zastoji)!

BURJA: Pic včasih napiše, da je burja 1. stopnje.

Stopnja 1
Zaradi burje je na vipavski hitri cesti med razcepom Nanos in priključkom Ajdovščina prepovedan promet za počitniške prikolice, hladilnike in vozila s ponjavami, lažja od 8 ton.

Stopnja 2
Zaradi burje je na vipavski hitri cesti med razcepom Nanos in Ajdovščino prepovedan promet za hladilnike in vsa vozila s ponjavami.

Preklic
Na vipavski hitri cesti in na regionalni cesti Ajdovščina - Podnanos ni več prepovedi prometa zaradi burje. 
Ali

Na vipavski hitri cesti je promet znova dovoljen za vsa vozila.

Do 21-ih velja prepoved prometa tovornih vozil, katerih največja dovoljena masa presega 7 ton in pol.
Od 8-ih do 21-ih velja prepoved prometa tovornih vozil, katerih največja dovoljena masa presega 7 ton in pol, na primorskih cestah ta prepoved velja do 22-ih.
"""

SYSTEM_PROMPT: str = f"""
Ti si profesionalen poročevalec prometnih poročil.
Tvoja naloga je oceniti generirana poročila.

Ta poročila so bila generirana na podlagi podatkov s spletne strani promet.si.
Poročila ne smejo vsebovati informacij, ki jih ni med podatki s spletne strani.
Poleg tega tudi ne smejo spuščati pomembnih podatkov.

Spodaj so podana navodila za kako mora biti poročilo ustvarjeno.
Tvoja naloga je številčno oceniti (od 1 do 10) dana poročila glede na to kako dobro sledijo tem navodilo in kako kvalitetna so na splošno.
10 pomeni, da je poročilo jedrnato, vsebuje vse zahtevane informacije in upošteva praktično vsa dana navodila.
5 pomeni, da je poročilo nekoliko predolgo, vsebuje kakšno informacijo preveč ali premalo, ali ne sledi navodilom zelo natančno.
1 pomeni, da sta vsebina s spletne strani in poročilo navidez nepovezana, ali pa je velika količina manjkajočih/odvečnih informacij, ali pa se sploh ne sledi danim navodilom.

Tvoj odziv mora biti v naslednjem formatu: "Ocena: x", kjer je x ocena od 1 do 10.
To mora biti celoten odziv, nič drugega.

Tukaj so navodila: {GENERATION_INSTRUCTIONS}

"""

# response_format = {
#     "name": "vrni_oceno_porocila",
#     "strict": True,
#     "schema": {
#         "type": "object",
#         "properties": {
#             "ocena_porocila": {
#                 "type": "number",
#                 "description": "Ocena danega poročila.",
#                 "minimum": 1,
#                 "maximum": 10
#             }
#         },
#         "additionalProperties": False,
#         "required": [
#             "ocena_porocila",
#         ]
#     }
# }

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

def get_score(user_prompt: str) -> float | None:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        # response_format=response_format,
    )
    response_str = completion.choices[0].message.content
    score = None
    try:
        iter = ONLY_NUMBERS.finditer(response_str)
        num = next(iter).group()
        if 1 <= float(num) <= 10:
            score = num
    except:
        pass
    return score

def main():
    user_prompt = "zelo dobro poročilo"
    print(get_score(user_prompt))

if __name__ == "__main__":
    main()
