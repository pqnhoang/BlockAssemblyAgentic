from .agent.block_design_mas import BlockDesignMAS
import warnings
warnings.filterwarnings('ignore')

designmas = BlockDesignMAS(api_file="api_key.txt", max_round=5)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    query = "Tree"
    positions='/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/final_results/positions/tree_result.json'
    structure_img = '/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/imgs/structures/tree/tree_isometric.png'
    save_path = designmas.query(query, positions = positions, structure_img=structure_img)