from enum import Enum

from distiller.filter.core import SentencePairHeuristicFilter
from distiller.filter.core import SentencePairModelFilter
 
class FILTER_DICT(Enum):
    SENTENCE_PAIR_HEURISTIC = SentencePairHeuristicFilter
    SENTENCE_PAIR_MODEL = SentencePairModelFilter
