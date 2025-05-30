import torch
import time
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft         import PeftConfig, PeftModel
from evaluation.llm_evaluation import GENERATION_INSTRUCTIONS

INPUTS_PATH: str = "data/dp1_inputs.jsonl"
OUTPUT_PATH: str = "data/dp1_outputs.jsonl"

# Path to your adapter
# PEFT_DIR = "/d/hpc/projects/onj_fri/peft_ah/2b-dp1"
# model_id = "cjvt/GaMS-2B"

# PEFT_DIR = "/d/hpc/projects/onj_fri/peft_ah/9b-instr-dp2"
# model_id = "cjvt/GaMS-9B-Instruct"

PEFT_DIR = "/d/hpc/projects/onj_fri/nmlp/PEFT"
model_id = "cjvt/GaMS-27B-Instruct"

model_to_dtype: dict[str, torch.dtype] = {
    "cjvt/GaMS-2B": torch.float32,
    "cjvt/GaMS-9B-Instruct": torch.bfloat16,
    "cjvt/GaMS-27B-Instruct": torch.bfloat16,
} 

shot_1: str = """
    VHOD:
    Prometne informacije       08. 08. 2024   	   11.30           2. program

    Podatki o prometu.

    Zastoji Na gorenjski avtocesti: - med Kranjem zahod in Kranjem vzhod proti Ljubljani, okvarjeno vozilo in delovna zapora - pred predorom Karavanke proti Avstriji, 500 m. Na cestah Lucija - Strunjan in Šmarje - Koper ter na vseh cestah, ki vodijo na Bled. Na dolenjski avtocesti: - predmet na vozišču ovira promet med Bičem in Ivančno Gorico proti Ljubljani. Zaradi poplavljenega vozišča je zaprta cesta Ptuj - Slovenska Bistrica pri železniškem podvozu na Pragerskem. Delo na cesti Popolne zapore: - Cesta Ruše - Puščava bo pri Fali zaprta do 9. avgusta zaradi obnove mostu. - Cesta Begunje - Bistrica med Slatno in Begunjami do 22. avgusta. Zaradi poplavljenega vozišča je zaprta cesta Ptuj - Slovenska Bistrica pri železniškem podvozu na Pragerskem. Več o delovnih zaporah v prometni napovedi . Buy vignette for Slovenia online Thererore long queues are expected in entering points from Austria to Slovenia, i.e. Karavanke tunnel (A2) and Sentilj/Spielfeld crossing (A1). Important reason for these queues is that drivers don't have vignette for Slovenian roads and have to buy them at the border. To reduce or even avoid long waiting periods drivers are strongly recommended to buy vignette for Slovenian motorways online. They can do it here . DARS Bernarda Kržičnik, PIC Prireditve  Tovorni promet   Opozorilo Na štajerski avtocesti pred prehodom Šentilj proti Avstriji, 1 km. Ovire Zaradi odstranjevanja okvarjenega vozila je zaprta cesta Strmec - Mangart, Cesta na Mangart . Zastoji Na gorenjski avtocesti: - med Kranjem zahod in Kranjem vzhod proti Ljubljani, okvarjeno vozilo in delovna zapora - pred predorom Karavanke proti Avstriji, 500 m. Zastoji Na gorenjski avtocesti: - pri priključku Jesenice vzhod proti Jesenicam, zaradi del; - pred predorom Karavanke proti Avstriji, 1,5 km. Zastoji Primorska: - Iz smeri Ljubljana Brdo ter Ljubljana Vič proti Brezovici, zamuda 10 minut; - iz Kopra proti Ljubljani med Logatcem proti Ljubljano, zamuda 30 minut. Na primorski avtocesti: - iz Ljubljane proti Brezovici; - iz Kopra proti Ljubljani med Logatcem proti Ljubljano, zamuda 15 minut. Na cestah Lucija - Strunjan in Šmarje - Koper ter na vseh cestah, ki vodijo na Bled. Na cestah Lucija - Strunjan in Šmarje - Koper ter na vseh cestah, ki vodijo na Bled. Zastoji Primorska: - Iz smeri Ljubljana Brdo ter Ljubljana Vič proti Brezovici, zamuda 10 minut; - iz Kopra proti Ljubljani med Logatcem proti Ljubljano, zamuda 30 minut. - Pred Šentiljem proti Avstriji 600 metrov. Na primorski avtocesti: - iz Ljubljane proti Brezovici, zamuda 10 minut; - iz Kopra proti Ljubljani med Logatcem proti Ljubljano, zamuda 25 minut.

    IZHOD:
    Prometne informacije      08.08.2024               11.30      2. program


    Podatki o prometu.

    Zastoji so na štajerski avtocesti med Trojanami in predorom Jasovnik proti Mariboru ter pred mejnim prehodom Šentilj; na gorenjski avtocesti pa pri priključku Jesenice-vzhod proti Karavankam in pred predorom Karavanke proti 
    Avstriji. Na primorski avtocesti nastajajo zastoji med Ljubljano in Brezovico proti Kopru, pa tudi v nasprotni smeri med Logatcem in Brezovico.

    Promet je zgoščen na cestah Lucija-Strunjan, Šmarje-Koper in Lesce-Bled.

    Zaradi odstranjevanja pokvarjenega vozila je zaprta cesta na Mangart pri Strmcu.
    
    <EOS>
"""

shot_2: str = """
    VHOD:
    Prometne informacije       05. 05. 2023   	   12.30           2. program

    Podatki o prometu.

    Delo na cesti Popolne zapore , dela: - Na gorenjski avtocesti sta zaprta priključka Brezje in Lesce proti Karavankam, predvidoma do 30. Zapora je predvidena do 30 maja. Nesreče Na primorski avtocesti med Kozarjami in Brezovico proti Kopru je zaprt prehitevalni pas. - Cesta Šmartno - Vodice - Brnik bo zaprta predvidoma do ponedeljka, 8. maja, do 5. ure mimo priključka Vodice zaradi gradnje obvoznice. Delo na cesti Popolne zapore , dela: - Na gorenjski avtocesti sta zaprta priključka Brezje in Lesce proti Karavankam, predvidoma do 30. ure. - Cesta Križaj - Čatež bo zaprta med Krško vasjo in Čatežem, od 8. do 12. Več o delovnih zaporah v prometni napovedi. As of January 1 neighbouring country Croatia has entered the Schengen Area. Border control between Slovenia and Croatia is lifted. There is no need to stop at borders. Drive carefully! Buy vignette for Slovenia online Thererore long queues are expected in entering points from Austria to Slovenia, i.e. Karavanke tunnel (A2) and Sentilj/Spielfeld crossing (A1). Important reason for these queues is that drivers don't have vignette for Slovenian roads and have to buy them at the border. To reduce or even avoid long waiting periods drivers are strongly recommended to buy vignette for Slovenian motorways online. They can do it here . DARS Barbara Janežič, PIC Prireditve Ovire Na štajerski avtocesti je oviran promet v predoru Trojane in za predorom Ločica proti Mariboru, okvare vozil. Na primorski avtocesti med Kozarjami in Brezovico proti Kopru so odstranili posledice prometne nesreče . Zastoji Zaradi popoldanske prometne konice nastajajo zastoji na cestah, ki vodijo iz mestnih središč, na mestnih obvoznicah še posebno na ljubljanski. Ovire Na primorski avtocesti je: - oviran promet med Brezovico in Kozarjami, predmet na vozišču; - zaprt počasni pas med Senožečami in Nanosom proti Ljubljani, okvara vozila. DARS Janja Budič, PIC Prireditve DARS Urh Šelih, PIC Prireditve Zastoji Na dolenjski avtocesti med priključkom Ivančna Gorica in Bič proti Novemu mestu. Vozniki, ki ste namenjeni proti Bledu, zapustite avtocesto na priključku Radovljica in se držite desnega pasu. DARS Bernarda Kržičnik, PIC Prireditve Nesreče Na zahodni ljubljanski obvoznici med Kosezami in Brdom proti Kozarjam je zaprt prehitevalni pas. Nastaja zastoj, ki sega proti Malencam in proti Kosezam. Ovire Predmet na vozišču ovira promet na: - severni ljubljanski obvoznici med Tomačevim in Zadobrovo proti Zadobrovi; - na ljubljanski obvoznici na razcepu Kozarje iz smeri Brezovice proti Viču. Nastaja zastoj, ki sega proti Malencam in proti Kosezam. Zamuda 10 minut. Zastoji Promet je zaradi popoldanske prometne konice povečan na cestah, ki vodijo iz mestnih središč, na mestnih obvoznicah še posebno na ljubljanski. Na štajerski avtocesti pred Pesnico proti Dragučovi.

    IZHOD:
    Prometne informacije        05. 05. 2023         12.30                2. program


    Zaradi pokvarjenega vozila je na primorski avtocesti proti Ljubljani zaprt pas za počasna vozila med priključkom Senožeče in razcepom Nanos.

    Na štajerski avtocesti proti Mariboru je zaradi del še vedno zastoj med razcepom Zadobrova in priključkom Šentjakob. Potovalni čas se podaljša za 10 minut.
    V isti smeri je zaradi pokvarjenega vozila promet oviran v predoru Trojane.

    <EOS>
"""

shot_3: str = """
    VHOD:
    Prometne informacije       04. 04. 2022   	   19.30           2. program

    Podatki o prometu.

    Opozorila Na primorski avtocesti je zaprt vozni pas skozi predor Dekani proti Ljubljani zaradi okvare tovornega vozila. Na primorski avtocesti je zaprt vozni pas skozi predor Dekani proti Ljubljani zaradi okvare tovornega vozila. Na primorski avtocesti je zaprt vozni pas skozi predor Dekani proti Ljubljani zaradi okvare tovornega vozila. Delo na cesti Dolenjska avtocesta bo danes ponoči od 20. do 5. ure zjutraj zaprta med priključkom Šmarje Sap in razcepom Malence proti Ljubljani zaradi nujnih vzdrževalnih del v predorih. Cesta čez prelaz Vršič je zaradi nevarnosti proženja snežnih plazov zaprta za ves promet. V sredo, 6. 4. aprila, bo na gorenjski avtocesti skozi predor Karavanke promet potekal izmenično enosmerno od 8. do 16. ure. Čakalna doba je na mejnih prehodih Obrežje in Gruškovje. V nočeh na sredo, četrtek in petek bo med 21. uro zaprta primorska avtocesta med Črnim Kalom in Srminom v obe smeri . Obvoz bo po regionalni cesti. DARS Barbara Janežič, PIC Na izvozu na počivališče so ustavljena tovorna vozila. Na primorski avtocesti je zaprt vozni pas skozi predor Dekani proti Ljubljani zaradi okvare tovornega vozila. Delo na cesti Dolenjska avtocesta bo do 5.

    IZHOD:
    Prometne informacije        04. 04. 2022       19.59         1. in 2. program

    Podatki o prometu.

    Na podravski avtocesti proti Mariboru je zaradi pokvarjenega vozila promet oviran med priključkoma Zlatoličje in Marjeta.

    Zaradi del bo do 5-ih zjutraj zaprta dolenjska avtocesta proti Ljubljani med priključkom Šmarje Sap in razcepom 
    Malence. Obvoz je po regionalni cesti čez Škofljico.

    Na mejnih prehodih Gruškovje in Obrežje vozniki tovornih vozil na vstop v državo čakajo približno 2 uri.
    <EOS>
"""

    # "The examples of antonyms are:\nhigh => low\nwide => narrow\nbig =>",
    # "Pristanek je bil prvi nadzorovani spust ameriškega vesoljskega plovila na površje Lune po Apollu 17 leta 1972, ko je na Luni pristala zadnja Nasina misija s posadko.\nDoslej so na Luni pristala vesoljska plovila le iz štirih drugih držav –",
    # "U četvrtak je bila prva polufinalna večer Dore, a komentari na društvenim mrežama ne prestaju. U nedjeljno finale prošli su:",

prompt_template = f"""
Si profesionalen poročevalec prometnih informacij.
Dobil boš nekaj podatkov, ki so bili pridobljeni s spletne strani prometnih informacij.
Tvoja naloga je da ustvariš kratko prometno poročilo (IZHOD) na podlagi teh podatkov (VHOD).
Poročilo se mora končati z oznako <EOS>

Oblika poročila:
------------------------------------------------------------
Prometne informacije       DD. MM. YYYY       HH.MM           2. program

Podatki o prometu.

<sledi povzetek> 
------------------------------------------------------------

Slediti moraš naslednjim navodilom:

### ZAČETEK NAVODIL

{GENERATION_INSTRUCTIONS}

### KONEC NAVODIL

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
""".strip()

example_input: str = """
Prometne informacije       09. 09. 2024      19.30           2. program

Podatki o prometu.

Nesreče Na podravski avtocesti je na razcepu Draženci iz smeri Ptuja proti Hrvaški oviran promet, na tem odseku je še spolzko vozišče. Previdno! Zastoji Na gorenjski avtocesti je zastoj tovornih vozil pred predorom Karavanke proti Avstriji, približno 3 kilometre. Proti Kranjski Gori je možen izvoz Jesenice vzhod/Lipce. Nesreče Na primorski avtocesti je pred Brezovico proti Kopru zaprt skrajno desni vozni pas. Na štajerski avtocesti med Framom in Polskavo v obe smeri, občasno, delovna zapora. Na cestah Lesce - Bled, Dobrova - Brezovica in Ljubljana - Brezovica. Zaradi popoldanske prometne konice je promet povečan na cestah iz mestnih središč in na mestnih obvoznicah. Delo na cesti Na gorenjski avtocesti v predoru Karavanke bo promet potekal izmenično enosmerno s čakalno dobo pred predorom: - danes med 14.30 in 20.30; - 10. in 11. september med 8. in 20.30 . Zastoj je tudi proti Sloveniji. Popolne zapore: - V noči s torka na sredo bo med 20. in 5. uro, na gorenjski avtocesti zaprt predor Šentvid proti Ljubljani. - V Zgornji Sorici od 10. septembra do 17. ure do 13. ure. Obvoz bo preko Petrovega Brda. Več o delovnih zaporah v prometni napovedi . Buy vignette for Slovenia online Thererore long queues are expected in entering points from Austria to Slovenia, i.e. Karavanke tunnel (A2) and Sentilj/Spielfeld crossing (A1). Important reason for these queues is that drivers don't have vignette for Slovenian roads and have to buy them at the border. To reduce or even avoid long waiting periods drivers are strongly recommended to buy vignette for Slovenian motorways online. They can do it here . DARS Nina Mesarič, PIC      Tovorna vozila Delo na cesti Na gorenjski avtocesti v predoru Karavanke bo promet potekal izmenično enosmerno s čakalno dobo pred predorom 10. DARS Barbara Janežič, PIC      Tovorna vozila

"""

def test():
    prompt = prompt_template.format(input=example_input)
    sequences = pline(
        [prompt],
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

def main():
    lines_to_skip = 0
    with open(OUTPUT_PATH, "rt") as file:
        lines_to_skip = len(file.readlines())
    print(f"Lines to skip: {lines_to_skip}")
    df: pd.DataFrame = pd.read_json(INPUTS_PATH, lines=True)
    example_inputs: list[str] = [x for x in df["text"]]
    example_inputs = example_inputs[lines_to_skip:]
    prompts: list[str] = [prompt_template.format(input=example) for example in example_inputs]
    with open(OUTPUT_PATH, "at") as file:
        for i in range(len(prompts)):
            print(f"[{i+1}/{len(prompts)}]")
            t0 = time.time()
            sequences = pline(
                [prompts[i]],
                max_new_tokens=512,
                num_return_sequences=1,
                return_full_text=False,
            )
            try:
                seq = sequences[0]
                result: str = seq[0]["generated_text"]
                result = result[:result.index("<EOS>")]
                file.write(json.dumps({"input": example_inputs[i], "output": result}, ensure_ascii=False) + "\n")
                file.flush()
            except Exception as e:
                print("Error while trying to save generated output: " + str(e))
            print(f"Took {time.time()-t0:.1f} seconds")
    print("Done")

if __name__ == "__main__":
    # 1) Load the adapter config to discover the base model name
    config = PeftConfig.from_pretrained(PEFT_DIR)
    base_name = config.base_model_name_or_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2) Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model     = AutoModelForCausalLM.from_pretrained(
        base_name,
        device_map=device,
        torch_dtype=model_to_dtype[model_id],
    )
    
    # 3) Wrap the base model with your PEFT adapter
    model = PeftModel.from_pretrained(
        model, PEFT_DIR,
        device_map=device,
        torch_dtype=model_to_dtype[model_id],
    )
    
    # pline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    pline = pipeline(
        "text-generation",
        model=model,
        device_map=device,
        torch_dtype=model_to_dtype[model_id],
        tokenizer=tokenizer,
    )
    main()

