# SimBinder-IF: Structure-Aware Antibody Affinity Optimization via Efficient Preference Learning

SimBinder-IF is an end-to-end framework for antibody affinity maturation. It leverages a structure-aware protein language model (ESM-IF) and an efficient preference learning strategy (SimPO/DPO) to optimize antibody binding affinity based on complex structures.

## 🚀 Key Features
- **Structure-Aware Scoring**: Uses the ESM-IF1 (GVP-based) backbone to capture 3D geometric constraints of antigen-antibody interfaces.
- **Preference Learning**: Implements SimPO (Simple Preference Optimization) and DPO (Direct Preference Optimization) for stable and efficient affinity optimization without an explicit reward model.
- **Multi-Chain Support**: Handles complex structures including heavy chains, light chains, and multi-chain antigens.
- **Antigen-Level Generalization**: Supports robust evaluation using antigen-based data splitting to ensure generalization to unseen targets.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA enabled GPU

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MSBMI-SAFE/SimBinder-IF.git
   cd SimBinder-IF
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install biotite pandas scikit-learn
   # Install ESM-IF official library
   pip install git+https://github.com/facebookresearch/esm.git
   ```

## 📂 Project Structure
- `train_esmif.py`: Main training script for preference-based fine-tuning.
- `data_utils.py`: Data pipeline for PDB structure processing and mutation sequence mapping.
- `util.py`: Utility functions for loss calculation (SimPO/DPO) and correlation metrics.

## 📊 Data Preparation
The pipeline expects a `metadata.json` file pointing to PDB structures and affinity CSVs. 
Each CSV should contain:
- `heavy_chain_seq`: Mutated antibody heavy chain sequence.
- `light_chain_seq`: Mutated antibody light chain sequence.
- `binding_score`: Measured affinity value (e.g., $K_d$, $\Delta G$).

## 🏋️ Training
To start fine-tuning with SimPO loss:
```bash
python train_esmif.py \
    --meta_json path/to/metadata.json \
    --out_dir checkpoints \
    --batch_size 32 \
    --lr 1e-4 \
    --beta 0.1 \
    --margin 0.1 \
    --higher_is_better True
```

## 📝 Citation
If you find this work helpful for your research, please cite our paper:
```bibtex
@article{simbinderif2026,
  title={SimBinder-IF: Structure-Aware Antibody Affinity Optimization via Efficient Preference Learning},
  author={...},
  journal={arXiv preprint},
  year={2026}
}
```
