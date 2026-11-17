"""Original model definitions; no alternative classifier or checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from glaucoma.models import ImageClassifier,build_model
from glaucoma.r23_model import DiscLocator
