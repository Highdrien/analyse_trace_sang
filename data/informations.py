import os
import sys
from os.path import dirname as up

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.labels import LABELS, BACKGROUND

DATAPATH = r'D:\LAC\2022_Projet Alicia'
# LABELS = ['1- Modèle Traces passives', '2- Modèle Goutte à Goutte', '3- Modèle Transfert par contact',
#           '4- Modèle Transfert glissé', '5- Modèle Altération par contact', '6- Modèle Altération glissée',
#           "7- Modèle d'Accumulation", '8- Modèle Coulée', '9- Modèle Chute de volume',
#           '10- Modèle Sang Propulsé', "11- Modèle d'éjection", '12- Modèle Volume impacté',
#           '13- Modèle Imprégnation', "14- Modèle Zone d'interruption", "15- Modèle Modèle d'impact",
#           "16- Modèle Foyer de modèle d'impact", '17- Modèle Trace gravitationnelle', '18- Modèle Sang expiré',
#           "19- Modèle Trace d'insecte"]
LABELS_PATH = list(map(lambda x: os.path.join(DATAPATH, x), LABELS))
IMAGE_SIZE = 128
# BACKGROUND = ['carrelage', 'papier', 'bois', 'lino']
DST_PATH = 'data/data_labo'