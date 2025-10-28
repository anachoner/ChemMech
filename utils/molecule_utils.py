from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string using RDKit
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if not smiles or not isinstance(smiles, str):
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        logger.warning(f"SMILES validation error: {e}")
        return False


def smiles_to_image(smiles: str, size=(300, 300)):
    """
    Convert SMILES string to molecule image
    
    Args:
        smiles: SMILES string
        size: Image size tuple (width, height)
    
    Returns:
        PIL Image object or None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES: {smiles}")
            return None
        
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        return img
        
    except Exception as e:
        logger.error(f"Error converting SMILES to image: {e}")
        return None


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES to canonical form
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        Canonical SMILES string
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.error(f"Error canonicalizing SMILES: {e}")
        return smiles