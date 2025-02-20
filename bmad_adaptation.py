import os
import shutil
import logging
from collections import defaultdict
import re
valid_paths = {
                "Brain_AD" : "valid/{label}/{type_dir}",
                "Chest_AD" : "val/{label}",
                "Histopathology_AD" : "valid/{label}",
                "Liver_AD" : "valid/{type_dir}/{label}",
                "Retina_OCT2017_AD" : "val/{label}",
                "Retina_RESC_AD" : "Val/{type_dir}/{label}"
              }
              
test_paths = {
               "Brain_AD" : "test/{label}/{type_dir}",
               "Chest_AD" : "test/{label}",
               "Histopathology_AD" : "test/{label}",
               "Liver_AD" : "test/{type_dir}/{label}",
               "Retina_OCT2017_AD" : "test/{label}",
               "Retina_RESC_AD" : "Test/{type_dir}/{label}"
             }
              
train_paths = {
                "Brain_AD" : "train/good",
                "Chest_AD" : "train/good",
                "Histopathology_AD" : "train/good",
                "Liver_AD" : "train/good",
                "Retina_OCT2017_AD" : "train/good",
                "Retina_RESC_AD" : "Train/train/good"
             }
             
image_level_class = ["Chest_AD", "Histopathology_AD", "Retina_OCT2017_AD"]

default_values = defaultdict(lambda : "")

# Configurazione del logging
logging.basicConfig(
    filename="restructure_log.txt",
    level=logging.INFO,
    filemode = "w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_dirs(base_path, dirs, dry_run):
    """ Crea le directory o simula l'operazione in modalità dry-run """
    for dir_name in dirs:

        dir_path = os.path.join(base_path, dir_name)
        if dry_run:
            logging.info(f"[DRY-RUN] Creata directory simulata: {dir_path}")
        else:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Creata directory: {dir_path}")

def move_files(src, dst, dry_run, add_mask_suffix = False):
    """ Sposta file o simula lo spostamento in modalità dry-run """
    if os.path.exists(src) and os.path.isdir(src):
       # print(f"Spostamento file da {src} a {dst}")
    
        for item in os.listdir(src):
            
            s = os.path.join(src, item)
            
            if add_mask_suffix:
                file_name, ext = os.path.splitext(item)
                d = os.path.join(dst, f"{file_name}_mask{ext}")
            else:
                d = os.path.join(dst, item)
            
            if dry_run:
                logging.info(f"[DRY-RUN] Spostato simulato: {s} -> {d}")
            else:
                try:
                    shutil.move(s, d)
                    logging.info(f"Spostato: {s} -> {d}")
                except Exception as e:
                    logging.error(f"Errore spostando {s} -> {d}: {e}")
    
def print_directory_tree(root_dir, level=0):
    """ Stampa l'anteprima visiva della struttura del dataset """
    print(" " * level * 4 + f"|-- {os.path.basename(root_dir)}")
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            print_directory_tree(item_path, level + 1)
        else:
            print(" " * (level + 1) * 4 + f"|-- {item}")
'''
def check_mapping(temp_path, mapping_dict):
    """Controlla se ci sono chiavi inutilizzate nel mapping tra i segnaposti del percorso e il dizionario contenente i valori delle mappature"""
    placeholders = re.findall(r'\{(\w+)\}', temp_path)
    ignored_keys = []
    
    for key in mapping_dict:
        if key not in placeholders:
            ignored_keys.append(key)
    return ignored_keys
'''

def restructure_dataset(root_dir, dry_run=False):
    """ Ristruttura il dataset con o senza dry-run """
    for category in os.listdir(root_dir):                           #Brain_AD, Liver_AD, Chest_AD...
        cat_path = os.path.join(root_dir, category)
        if not os.path.isdir(cat_path):
            continue

        for dataset in os.listdir(cat_path):
            dataset_path = os.path.join(cat_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            logging.info(f"Inizio ristrutturazione per: {dataset_path} (Dry-run: {dry_run})")

            # Crea la nuova struttura al PRIMO LIVELLO
            create_dirs(cat_path, ["ground_truth", "test/Ungood", "test/good", "train/good"], dry_run)
            #print(f"=================Inizio immagini e maschere testing per {category}\n")
           
            #Sposta le immagini di testing e le relative maschere (se presenti)
            for type_label in ["good", "Ungood"]:
                type_direcories = (["img", "label"] if "{type_dir}" in test_paths[category] and category != "Retina_RESC_AD" else ["test", "test_label"] if category == "Retina_RESC_AD" else [""])
                
                for type_directory in type_direcories:
                    test_src = os.path.join(dataset_path, test_paths[category].format_map({**default_values, "label" : type_label, "type_dir" : type_directory}))
                    '''
                    if os.path.exists(test_src):
                        print(test_src)
                    '''
                    if type_directory == "img" or type_directory == "test" or type_directory == "":
                        if type_label == "good":
                            move_files(test_src, os.path.join(cat_path, "test/good"), dry_run, add_mask_suffix = False)
                        else:
                            move_files(test_src, os.path.join(cat_path, "test/Ungood"), dry_run, add_mask_suffix = False)
                    
                    elif type_directory == "label" or type_directory == "test_label" and category not in image_level_class:
                            if type_label == "Ungood":
                                move_files(test_src, os.path.join(cat_path, "ground_truth"), dry_run, add_mask_suffix = True)
            
            #print(f"=================Fine immagini e maschere testing per {category}\n")
            #print(f"=================Inizio immagini e maschere validation per {category}\n")
            #Sposta le immagini di validation e le relative maschere (se presenti)
            for type_label in ["good", "Ungood"]:
                type_direcories = (["img", "label"] if "{type_dir}" in valid_paths[category] and category != "Retina_RESC_AD" else ["val", "val_label"] if category == "Retina_RESC_AD" else [""])
                
                for type_directory in type_direcories:
                    valid_src = os.path.join(dataset_path, valid_paths[category].format_map({**default_values, "label" : type_label, "type_dir" : type_directory}))
                    '''
                    if os.path.exists(test_src):
                        print(test_src)
                    '''
                    if type_directory == "img" or type_directory == "val" or type_directory == "":
                        if type_label == "good":
                            move_files(valid_src, os.path.join(cat_path, "train/good"), dry_run, add_mask_suffix = False)
                        else:
                            move_files(valid_src, os.path.join(cat_path, "test/Ungood"), dry_run, add_mask_suffix = False)
                    
                    elif type_directory == "label" or type_directory == "val_label" and category not in image_level_class:
                            if type_label == "Ungood":
                                move_files(valid_src, os.path.join(cat_path, "ground_truth"), dry_run, add_mask_suffix = True)
            
            #print(f"=================Fine immagini e maschere validation per {category}\n")
            
            #print(f"=================Inizio immagini di train per {category}\n")
            #Sposta le immagini di training (se presenti)
            train_src = os.path.join(dataset_path, train_paths[category])
            move_files(train_src, os.path.join(cat_path, "train/good"), dry_run)
            #print(f"=================Fine immagini di train per {category}\n")
            #Elimina la directory del secondo livello
            if not dry_run:
                try:
                    shutil.rmtree(dataset_path)
                    logging.info(f"Rimossa directory del secondo livello: {dataset_path}")
                except Exception as e:
                    logging.error(f"Errore eliminando {dataset_path}: {e}")
            else:
                logging.info(f"[DRY-RUN] Simulata eliminazione di: {dataset_path}")
                                
                            
                    
if __name__ == "__main__":
    root_directory = r"/mnt/mydisk/manuel_barusco/datasets/bmad"  # Modifica il percorso se necessario

    # Modalità Dry-Run Attivata (cambia a False per eseguire realmente) 
    dry_run_mode = False
    
    #print("\nPrima della Ristrutturazione:")
    #print_directory_tree(root_directory)
    
    restructure_dataset(root_directory, dry_run=dry_run_mode)
    
    #print("\nPrima della Ristrutturazione:")
    #print_directory_tree(root_directory)
