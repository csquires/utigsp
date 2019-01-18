from real_data_analysis.dixit.create_significant_effect_list import ivs2significant_effects
from real_data_analysis.dixit.dixit_meta import nnodes

npossible_effects = len(ivs2significant_effects)*(nnodes-1)
npositives = sum(len(effects) for iv_nodes, effects in ivs2significant_effects.items())
