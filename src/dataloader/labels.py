import torch

LABELS = [
    "1- Modèle Traces passives",
    "2- Modèle Goutte à Goutte",
    "3- Modèle Transfert par contact",
    "4- Modèle Transfert glissé",
    "5- Modèle Altération par contact",
    "6- Modèle Altération glissée",
    "7- Modèle d'Accumulation",
    "8- Modèle Coulée",
    "9- Modèle Chute de volume",
    "10- Modèle Sang Propulsé",
    "11- Modèle d'éjection",
    "12- Modèle Volume impacté",
    "13- Modèle Imprégnation",
    "14- Modèle Zone d'interruption",
    "15- Modèle Modèle d'impact",
    "16- Modèle Foyer de modèle d'impact",
    "17- Modèle Trace gravitationnelle",
    "18- Modèle Sang expiré",
]
BACKGROUND = ["carrelage", "papier", "bois", "lino"]


def get_label_prediction(y_pred: torch.Tensor) -> tuple[int, str]:
    if y_pred.shape != (len(LABELS)):
        raise ValueError(f'Expected y_pred with shape of {len(LABELS)}'
                         f'but found {y_pred.shape}')
    
    pred = torch.argmax(y_pred)
    return (pred, LABELS.index(pred))