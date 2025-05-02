from transformers import pipeline
import torch

# model_id = "cjvt/GaMS-2B"
# model_id = "cjvt/GaMS-9B-Instruct"
model_id = "cjvt/GaMS-27B-Instruct"

pline = pipeline(
    "text-generation",
    model=model_id,
    # device_map="cuda",
    device_map="auto", # Multi-GPU
    model_kwargs={"torch_dtype": torch.bfloat16},
)

shot_1: str = """
    VHOD:
    <p><strong>Zastoji</strong></p>
    <p>- na ljubljanski južni obvoznici od Rudnika proti razcepu Kozarje;
        <br>- na gorenjski avtocesti od Kosez do predora Šentvid proti Kranju;
        <br>- na štajerski avtocesti med Blagovico in Trojanami proti Celju ter pred prehodom Šentilj proti Avstriji;
        <br>- na ljubljanski zahodni obvoznici od Kosez proti Kozarjam ter na primorski avtocesti med Ljubljano in Brezovico proti Kopru;
        <br>- na cestah Ljubljana - Brezovica, Lesce - Bled in Šmarje - Koper.
    </p>

    IZHOD:
    Zastoji so na cestah Šmarje-Koper in Lesce-Bled.
    Prav tako so zastoji na zahodni in južni ljubljanski obvoznici od Kosez in Rudnika proti Primorski. Potovalni čas se na obeh odsekih podaljša za približno 15 minut.
    Zaradi del je krajši zastoj na štajerski avtocesti med Blagovico in Trojanami proti Mariboru. Opozarjamo na nevarnost naleta.
"""

shot_2: str = """
    VHOD:
    <p><strong>Zastoji</strong></p>
    <p>Na štajerski avtocesti:
        <br>- med Lukovico in Blagovico proti Celju, zamuda dobre pol ure
    </p>
    <p>Na regionalni cesti Blagovica - Trojane - Vransko.</p>
    <p><strong>Avtoceste:</strong>
        <br><strong>- Štajerska avtocesta bo zaprta med priključkoma Blagovica in Vransko proti Mariboru do ponedeljka, 2. 12. do 5. ure. Na priključku Trojane sta zaprta tudi uvoza proti Mariboru in Ljubljani, obvoz je urejen po regionalni cesti. Pričakujemo zastoje.</strong>
        <br><strong>- </strong>Na štajerski avtocesti med Framom in Slovensko Bistrico bodo do nedelje, 1. 12., izmenično zapirali vozni in prehitevalni pas v obe smeri zaradi odstranjevanja delovne zapore. Lahko nastane zastoj.
        <br>- Na primorski avtocesti med Logatcem in Uncem proti Kopru promet poteka samo po prehitevalnem pasu.
    </p>
    <p><strong>Popolne zapore na ostalih cestah:</strong>
        <br>- Cesta Koprivna - Črna, pri Pristavi, bo zaprta do 17. ure.
        <br>- Cesta Litija - Zagorje bo zaprta pri Šklendrovcu danes do 17. ure ter v prihodnjem tednu od ponedeljka do sobote, vsak dan med 8. in 17. uro.
        <br>- Cesta Bistrica pri Tržiču -Begunje, v Bistrici, na Begunjski cesti, do nedelje do 17. ure.
        <br>- V Ljubljani bo zaprta Litijska cesta med Pesarsko cesto in Potjo na Breje danes od 8. ure do polnoči.
    </p>

    IZHOD:
    Zastoji so na štajerski avtocesti med Lukovico in Blagovico proti Celju, čas vožnje se podaljša za dobre pol ure. Zastoji so tudi na regionalni cesti Blagovica - Trojane - Vransko.
    Pokvarjeno vozilo ovira promet na štajerski avtocesti pred priključkom Celje zahod proti Mariboru.
    Štajerska avtocesta bo zaprta med priključkoma Blagovica in Vransko proti Mariboru do ponedeljka do 5. ure. Na priključku Trojane sta zaprta tudi uvoza proti Mariboru in Ljubljani, obvoz je urejen po regionalni cesti.
    Na štajerski avtocesti med Framom in Slovensko Bistrico izmenično zapirajo vozni in prehitevalni pas v obe smeri zaradi odstranjevanja delovne zapore.
    Na primorski avtocesti med Logatcem in Uncem proti Kopru promet poteka samo po prehitevalnem pasu.
    Cesta Koprivna - Črna, pri Pristavi, bo zaprta do 17. ure, prav tako cesta Litija - Zagorje pri Šklendrovcu in cesta Bistrica pri Tržiču - Begunje, v Bistrici, na Begunjski cesti
    V Ljubljani bo do polnoči zaprta Litijska cesta med Pesarsko cesto in Potjo na Breje.
"""

shot_3: str = """
    VHOD:
    <p><strong>Nesreče</strong></p>
    <p>Na štajerski avtocesti je zaprt vozni pas med priključkoma Vransko in Šentrupert proti Mariboru. Nastal je zastoj, zamuda 10 - 15 minut.</p>
    <p><strong>Ovire</strong></p>
    <p>Na gorenjski avtocesti oviran promet zaradi okvare tovornega vozila med predorom Ljublbno in priključkom Podtabor proti Ljubljani.</p>
    <p><strong>Delo na cesti</strong></p>
    <p>Na gorenjski avtocesti bo promet med 11. in 16. uro potekal izmenično enosmerno skozi predor Karavanke.</p>

    IZHOD:
    Na štajerski avtocesti proti Mariboru je zaradi nesreče zaprt vozni pas med priključkoma Vransko in Šentrupert. Nastal je krajši zastoj. Opozarjamo na nevarnost naleta.
    Proti Ljubljani je na istem odseku zaradi pokvarjenega vozila oviran promet pred predorom Ločica.
    Pokvarjeno vozilo ovira promet tudi na gorenjski avtocesti med predorom Ljubno in priključkom Podtabor proti Ljubljani.
    Od 11-ih do 16-ih bo promet skozi predor Karavanke zaradi del urejen izmenično enosmerno.
"""

    # "The examples of antonyms are:\nhigh => low\nwide => narrow\nbig =>",
    # "Pristanek je bil prvi nadzorovani spust ameriškega vesoljskega plovila na površje Lune po Apollu 17 leta 1972, ko je na Luni pristala zadnja Nasina misija s posadko.\nDoslej so na Luni pristala vesoljska plovila le iz štirih drugih držav –",
    # "U četvrtak je bila prva polufinalna večer Dore, a komentari na društvenim mrežama ne prestaju. U nedjeljno finale prošli su:",
prompts = [
    f"""
    Si profesionalen poročevalec prometnih informacij.
    Dobil boš nekaj podatkov, ki so bili pridobljeni s spletne strani prometnih informacij.
    Tvoja naloga je da ustvariš kratko prometno poročilo (IZHOD) na podlagi teh podatkov (VHOD).

    Nadaljuj zaporedje naslednjih primerov:

    {shot_1}

    ### NASLEDNJI PRIMER ###

    {shot_2}

    ### NASLEDNJI PRIMER ###

    {shot_3}

    ### NASLEDNJI PRIMER ###

    VHOD:
    {{input}}

    IZHOD:
    """,
]

input: str = """
    <p><strong>Ovire</strong></p>
    <p>Na štajerski avtocesti je zaradi okvare vozila oviran promet na zaviralnem pasu pred izvozom Blagovica iz smeri Ljubljane.</p>
    <p>Na primorski avtocesti je zaradi okvare vozila oviran promet med razcepom Nanos in priključkom Senožeče proti Kopru.</p>
    <p>Na gorenjski avtocesti je zaradi okvare vozila oviran promet med priključkom Lesce in galerijo Moste proti Karavankam.</p>
    <p><strong>Tovorna vozila</strong></p>
    <p><strong>Omejitev prometa tovornih vozil, katerih največja dovoljena masa presega 7,5 t:</strong>
        <br>- danes, 31. 3., do 22. ure;
        <br>- v ponedeljek, 1. 4., med 8. in 22. uro.
    </p>
    <p><strong>Opozorila</strong></p>
    <p>V ponedeljek, 1. 4., in torek, 2. 4., je pričakovati povečan promet iz smeri Hrvaške proti notranjosti Slovenije ter naprej proti Avstriji in Italiji.</p>
    <p><strong>Delo na cesti</strong></p>
    <p><strong>Popolne zapore na priključkih:</strong>
        <br>- na ljubljanski severni obvoznici izvoz Podutik iz smeri Kosez;
        <br>- na gorenjski avtocesti, uvoz Šentvid s Celovške ceste proti Kosezam in uvoz Podutik proti Karavankam.
    </p>
    <p>V Domžalah je zaprta Virska cesta, med Ljubljansko cesto in Podrečjem. Do 10. aprila bo promet preko priključka Domžale proti Kamniku preusmerjen na lokalne ceste.</p>
"""

prompts = [prompt.format(input=input) for prompt in prompts]

sequences = pline(
    prompts,
    max_new_tokens=512,
    num_return_sequences=1
)

for seq in sequences:
    print("--------------------------")
    print(f"Result: {seq[0]['generated_text']}")
    print("--------------------------\n")

