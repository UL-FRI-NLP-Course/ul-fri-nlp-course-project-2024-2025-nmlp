from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel, PeftConfig
from transformers import pipeline
import torch

PEFT_DIR = "/d/hpc/projects/onj_fri/peft_ah/2b-dp1" # change according to the model
model_id = "cjvt/GaMS-2B"  # Change to "cjvt/GaMS-9B-Instruct" or "cjvt/GaMS-27B-Instruct" as needed

# Load PEFT config to get base model
config = PeftConfig.from_pretrained(PEFT_DIR)
base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, PEFT_DIR)
tokenizer = AutoTokenizer.from_pretrained(PEFT_DIR)

model_to_dtype: dict[str, torch.dtype] = {
    "cjvt/GaMS-2B": torch.float32,
    "cjvt/GaMS-9B-Instruct": torch.bfloat16,
    "cjvt/GaMS-27B-Instruct": torch.bfloat16,
} 


# pline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
pline = pipeline(
    "text-generation",
    model=model_id,
    # device_map="cuda",
    device_map="auto", # Multi-GPU
    torch_dtype=model_to_dtype[model_id],
)


shot_1: str = """
    VHOD:
    na ljubljanski južni obvoznici od Rudnika proti razcepu Kozarje. na gorenjski avtocesti od Kosez do predora Šentvid proti Kranju. na štajerski avtocesti med Blagovico in Trojanami proti Celju ter pred prehodom Šentilj proti Avstriji. na ljubljanski zahodni obvoznici od Kosez proti Kozarjam ter na primorski avtocesti med Ljubljano in Brezovico proti Kopru. na cestah Ljubljana - Brezovica, Lesce - Bled in Šmarje - Koper.

    IZHOD:
    Zastoji so na cestah Šmarje-Koper in Lesce-Bled.
    Prav tako so zastoji na zahodni in južni ljubljanski obvoznici od Kosez in Rudnika proti Primorski. Potovalni čas se na obeh odsekih podaljša za približno 15 minut.
    Zaradi del je krajši zastoj na štajerski avtocesti med Blagovico in Trojanami proti Mariboru. Opozarjamo na nevarnost naleta.
    <EOS>
"""

shot_2: str = """
    VHOD:
    Na štajerski avtocesti:
        med Lukovico in Blagovico proti Celju, zamuda dobre pol ure.
    Na regionalni cesti Blagovica - Trojane - Vransko.
        Štajerska avtocesta bo zaprta med priključkoma Blagovica in Vransko proti Mariboru do ponedeljka, 2. 12. do 5. ure. Na priključku Trojane sta zaprta tudi uvoza proti Mariboru in Ljubljani, obvoz je urejen po regionalni cesti. Pričakujemo zastoje.
        Na štajerski avtocesti med Framom in Slovensko Bistrico bodo do nedelje, 1. 12., izmenično zapirali vozni in prehitevalni pas v obe smeri zaradi odstranjevanja delovne zapore. Lahko nastane zastoj.
        Na primorski avtocesti med Logatcem in Uncem proti Kopru promet poteka samo po prehitevalnem pasu.
    Popolne zapore na ostalih cestah:
        Cesta Koprivna - Črna, pri Pristavi, bo zaprta do 17. ure.
        Cesta Litija - Zagorje bo zaprta pri Šklendrovcu danes do 17. ure ter v prihodnjem tednu od ponedeljka do sobote, vsak dan med 8. in 17. uro.
        Cesta Bistrica pri Tržiču -Begunje, v Bistrici, na Begunjski cesti, do nedelje do 17. ure.
        V Ljubljani bo zaprta Litijska cesta med Pesarsko cesto in Potjo na Breje danes od 8. ure do polnoči.

    IZHOD:
    Zastoji so na štajerski avtocesti med Lukovico in Blagovico proti Celju, čas vožnje se podaljša za dobre pol ure. Zastoji so tudi na regionalni cesti Blagovica - Trojane - Vransko.
    Pokvarjeno vozilo ovira promet na štajerski avtocesti pred priključkom Celje zahod proti Mariboru.
    Štajerska avtocesta bo zaprta med priključkoma Blagovica in Vransko proti Mariboru do ponedeljka do 5. ure. Na priključku Trojane sta zaprta tudi uvoza proti Mariboru in Ljubljani, obvoz je urejen po regionalni cesti.
    Na štajerski avtocesti med Framom in Slovensko Bistrico izmenično zapirajo vozni in prehitevalni pas v obe smeri zaradi odstranjevanja delovne zapore.
    Na primorski avtocesti med Logatcem in Uncem proti Kopru promet poteka samo po prehitevalnem pasu.
    Cesta Koprivna - Črna, pri Pristavi, bo zaprta do 17. ure, prav tako cesta Litija - Zagorje pri Šklendrovcu in cesta Bistrica pri Tržiču - Begunje, v Bistrici, na Begunjski cesti
    V Ljubljani bo do polnoči zaprta Litijska cesta med Pesarsko cesto in Potjo na Breje.
    <EOS>
"""

shot_3: str = """
    VHOD:
    Na štajerski avtocesti je zaprt vozni pas med priključkoma Vransko in Šentrupert proti Mariboru. Nastal je zastoj, zamuda 10 - 15 minut.
    Na gorenjski avtocesti oviran promet zaradi okvare tovornega vozila med predorom Ljublbno in priključkom Podtabor proti Ljubljani.
    Delo na cesti.
    Na gorenjski avtocesti bo promet med 11. in 16. uro potekal izmenično enosmerno skozi predor Karavanke.

    IZHOD:
    Na štajerski avtocesti proti Mariboru je zaradi nesreče zaprt vozni pas med priključkoma Vransko in Šentrupert. Nastal je krajši zastoj. Opozarjamo na nevarnost naleta.
    Proti Ljubljani je na istem odseku zaradi pokvarjenega vozila oviran promet pred predorom Ločica.
    Pokvarjeno vozilo ovira promet tudi na gorenjski avtocesti med predorom Ljubno in priključkom Podtabor proti Ljubljani.
    Od 11-ih do 16-ih bo promet skozi predor Karavanke zaradi del urejen izmenično enosmerno.
    <EOS>
"""

    # "The examples of antonyms are:\nhigh => low\nwide => narrow\nbig =>",
    # "Pristanek je bil prvi nadzorovani spust ameriškega vesoljskega plovila na površje Lune po Apollu 17 leta 1972, ko je na Luni pristala zadnja Nasina misija s posadko.\nDoslej so na Luni pristala vesoljska plovila le iz štirih drugih držav –",
    # "U četvrtak je bila prva polufinalna večer Dore, a komentari na društvenim mrežama ne prestaju. U nedjeljno finale prošli su:",
prompts = [
    f"""
    Poziv za generiranje prometnih poročil (RTV Slovenija)
    Generiraj kratko in strukturirano prometno poročilo v slogu RTV Slovenija na podlagi tabelaričnih prometnih dogodkov (iz preteklih X ur, npr. 3–4 ure).
    Oblika poročila:
    ------------------------------------------------------------
    Prometne informacije       DD. MM. YYYY       HH.MM           2. program
    
    Podatki o prometu.
    
    <sledi povzetek> 
    ------------------------------------------------------------
    
    Navodila za vsebino:
    - Povzemi pomembne dogodke: nesreče, zapore cest, živali na vozišču, izredne razmere (burja, tovorna prepoved).
    - Vključi strukturo stavkov:
        1. opcija: Cesta in smer + razlog + posledica + odsek
        2. opcija: Razlog + cesta in smer + posledica + odsek
    - Prednost imajo stavki, ki vključujejo:
      - znana cestna imena (avtoceste, razcepi, mesta),
      - konkretne učinke na promet (zastoji, zapora pasu),
      - točne lokacije (med priključkoma X in Y).
    
    Navodila za slog:
    - Ne ponavljaj istih ali zelo podobnih informacij.
    - Združi podobne stavke, obdrži najbolj informativen.
    - Ne vključuj trivialnih informacij ali običajnih konic, razen če so izredno dolge ali nenavadne.
    - Vsak stavek naj bo jedrnat in razumljiv za branje v etru.
    
    Poročilo se mora končati z oznako <EOS>
    Spodaj sledijo primeri vhodnih podatkov (iz Excela) in ustrezno oblikovanih izhodnih poročil:

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
    f"""
    Si profesionalen poročevalec prometnih informacij.
    Dobil boš nekaj podatkov, ki so bili pridobljeni s spletne strani prometnih informacij.
    Tvoja naloga je da ustvariš kratko prometno poročilo (IZHOD) na podlagi teh podatkov (VHOD).
    Poročilo se mora končati z oznako <EOS>

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
    Na štajerski avtocesti je zaradi okvare vozila oviran promet na zaviralnem pasu pred izvozom Blagovica iz smeri Ljubljane.
    Na primorski avtocesti je zaradi okvare vozila oviran promet med razcepom Nanos in priključkom Senožeče proti Kopru.
    Na gorenjski avtocesti je zaradi okvare vozila oviran promet med priključkom Lesce in galerijo Moste proti Karavankam.
    Omejitev prometa tovornih vozil, katerih največja dovoljena masa presega 7,5 t:
        danes, 31. 3., do 22. ure;
        v ponedeljek, 1. 4., med 8. in 22. uro.
    V ponedeljek, 1. 4., in torek, 2. 4., je pričakovati povečan promet iz smeri Hrvaške proti notranjosti Slovenije ter naprej proti Avstriji in Italiji.
    Delo na cesti
    Popolne zapore na priključkih:
        na ljubljanski severni obvoznici izvoz Podutik iz smeri Kosez;
        na gorenjski avtocesti, uvoz Šentvid s Celovške ceste proti Kosezam in uvoz Podutik proti Karavankam.
    V Domžalah je zaprta Virska cesta, med Ljubljansko cesto in Podrečjem. Do 10. aprila bo promet preko priključka Domžale proti Kamniku preusmerjen na lokalne ceste.
"""

prompts = [prompt.format(input=input) for prompt in prompts]

sequences = pline(
    prompts,
    max_new_tokens=512,
    num_return_sequences=1,
    return_full_text=False,
)


for seq in sequences:
    result: str = seq[0]["generated_text"]
    try:
        result = result[:result.index("<EOS>")]
    except:
        print("Error while trying to cut generated output")

    print("--------------------------")
    print(f"Result: {result}")
    print("--------------------------\n")



