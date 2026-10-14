# P-Rob3 thesis — Pulizia del progetto senza interrompere OpenVLA

## Stato e obiettivo

Questo progetto comprende tre pipeline distinte: **robosuite/HDF5** (raccolta dati), **TFDS/RLDS** (preparazione dati) e **OpenVLA** (fine-tuning e inferenza). La riorganizzazione serve a renderle documentabili e riproducibili, non a cambiare l'esperimento già in corso.

Durante il fine-tuning **non spostare e non cancellare** `data/rlds`, `runs/openvla`, `adapters/openvla`, `cache/huggingface`, `wandb`, `integrations/openvla`, `tools`, `openvla_env`, `venv`, `scripts/data`, `assets` o il codice operativo in `scripts`. Non cambiare i loro nomi e non riavviare il training solo per fare ordine. Evita comandi `find`/`du` molto pesanti su tutta la directory mentre le GPU lavorano.

## Pulizia minima, sicura e reversibile, ora

Dalla root `~/Scrivania/thesis`:

```bash
python maintenance/organize_project.py audit
python maintenance/organize_project.py safe --apply
```

Lo script crea `docs/guides`, `docs/architecture`, `archive/legacy`, `data/splits` e, **solo se non sono referenziati da altri script locali**, archivia `scripts/export_lerobot_to_rlds_raw.py` e sposta `structure.txt` in `docs/structure_original.txt`. Non sposta il builder vecchio, non cambia import e non tocca percorsi usati dal training. Il resoconto reversibile è `archive/cleanup_moves.json`.

Rollback:

```bash
python maintenance/organize_project.py restore
```

Non cancella cartelle esistenti. Le cartelle nuove vuote rimangono dopo il rollback.

## Distinzione fra i builder

* **Attivo HDF5 → RLDS:** `tools/rlds_dataset_builder/prob3_pick_place/prob3_pick_place_dataset_builder.py` (legge `PROB3_HDF5_ROOT`, h5py, `actions_openvla`).
* **Storico LeRobot → NPY → RLDS:** `rlds_dataset_builder/prob3_pick_and_place/prob3_pick_place_dataset_builder.py` (legge `episode_*.npy`).

Non eliminare né rinominare il secondo automaticamente: prima controlla con una ricerca dei riferimenti e con git che non venga utilizzato da altri comandi/esperimenti. Per il normale workflow usa soltanto il builder attivo.

## Dopo il training: fase strutturale (fare in commit Git separato)

1. Assicurati che il fine-tuning sia terminato e che i checkpoint abbiano i relativi `dataset_statistics.json`, `adapter_config.json` e file di peso.
2. Fai `git status` e un backup/commit delle modifiche. Conserva i dataset originali e non committare checkpoint o HDF5 enormi nel Git standard.
3. Aggiorna i percorsi in `integrations/openvla/register_prob3.py`, `train_prob3_lora.sh`, `finetune_prob3.py`, `preflight_prob3.py`, e in ogni comando/guida prima di spostare directory.
4. Per `scripts/`, **non fare un `mv` diretto**: i moduli sono importati per nome e `custom_robot.py` calcola i percorsi MJCF in base a `__file__` (`../assets`). Una migrazione in `src` richiede una sistemazione esplicita di package e asset resolution e successivi test di avvio della simulazione.
5. Se vuoi spostare gli HDF5 da `scripts/data` a `data/raw`, attenzione agli HDF5 con collegamenti esterni: vanno ricreati/aggiornati i link e mantenuti i sorgenti finché il reader non è stato verificato.
6. Mantieni `tools/openvla` e `tools/rlds_dataset_builder` come repository di terze parti; scrivi modifiche e adattatori in `integrations/` invece di modificare silenziosamente le dipendenze.
7. Congela un manifest dell'esperimento: commit/versione OpenVLA, configurazione OSC, seed e split episodio, percorsi HDF5, versione builder, parametri LoRA, checkpoint e statistiche normalizzazione.
8. Solo dopo verifica robosuite → HDF5 → RLDS → preflight → inferenza (una demo) prima di rimuovere il vecchio codice.

## Struttura-obiettivo (dopo i test)

```text
thesis/
├── assets/                     # MJCF, mesh, texture
├── src/                        # Codice tuo, importabile come package
│   └── prob3/
│       ├── simulation/
│       ├── control/
│       └── recording/
├── integrations/
│   └── openvla/                # Registrazione, conversione, training, eval
├── maintenance/                # Audit e utility organizzative
├── data/
│   ├── raw/                    # HDF5 sorgenti
│   ├── rlds/                   # Dataset attivo (eventualmente versionato)
│   └── splits/                 # Manifest episodio train/val/test
├── runs/                       # Logs e checkpoint, esclusi da git
├── adapters/                   # LoRA, esclusi da git
├── docs/                       # Guide, protocollo, risultati
├── tools/                      # Repository terze parti
├── archive/                    # Vecchi script
└── README.md
```

## Interpretazione del timer

Una barra `100/8000 [01:00<01:19:00, 1.67it/s]` significa circa un minuto trascorso, 79 minuti stimati rimanenti e 1,67 step al secondo. Il tempo è una stima, spesso non tiene conto dei picchi di I/O e merge durante il checkpoint. Durante il training non avviare conversioni o scansioni su grandi HDF5 in parallelo.
