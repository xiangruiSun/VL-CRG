# VL-CRG: Vision-Language Model with Causal Relation Graph Pre-Training

This repository extends **LXMERT** to incorporate **Causal Relation Graph (CRG) Pre-Training** for **Visual Commonsense Reasoning (VCR)** and **Visual Question Answering (VQA)**. It enhances structured reasoning by learning causal interactions between objects and textual elements, improving interpretability and robustness.

This project is based on [LXMERT](https://github.com/airsplay/lxmert), originally designed for vision-language tasks. We introduce modifications to integrate **causal reasoning** and **structured learning**.

---

## 🔥 Key Contributions
- **Causal Pre-Training (CRG-PT):** Models causal dependencies in vision-language data.
- **Structured Reasoning:** Learns object interactions via **Causal Relation Graphs (CRGs)**.
- **Improved Robustness:** Evaluates **Consistency** and **Sensitivity** in VCR/VQA tasks.
- **Performance Gains:** Achieves **higher accuracy**, **better generalization**, and **faster convergence**.

---

## 📌 Pre-trained Models
The pre-trained **LXMERT** model (870 MB) is available for download:

```bash
mkdir -p snap/pretrained
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```

For our **VL-CRG Pre-trained Model**, please check our **model release page**.

---

## 🚀 Running VL-CRG on VCR/VQA

### 1️⃣ **Setup Environment**
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ **Download Data**
We use **MS COCO** images and **VCR/VQA** annotations.

#### **Image Features (MS COCO)**
```bash
mkdir -p data/mscoco_imgfeat
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip

wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
```

#### **Convert Image Features**
```bash
python src/tools/detection_feature_converter.py
```

#### **Process Question-Answer Annotations**
```bash
python src/tools/compute_softscore.py 
```

---

### 3️⃣ **Fine-Tuning VL-CRG on VCR/VQA**
Modify `src/config.py` to specify dataset.

Run fine-tuning:
```bash
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py \
--train train --valid val \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --optim bert --lr 3e-5 --epochs 20 \
--name vl-crg-vqa
```

---

### 4️⃣ **Evaluating VL-CRG**
```bash
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py \
--train train --test val \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --load output/vl-crg-vqa.pth \
--tqdm
```

Check accuracy breakdown:
```bash
python acc_per_type.py output/val_predict.json
```

---

### 5️⃣ **Fine-Tuning VL-CRG on NLVR²**
Modify `src/config.py` to set dataset paths and task as `nlvr2`.

Run fine-tuning:
```bash
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/nlvr2.py \
--train train --valid val \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --optim bert --lr 3e-5 --epochs 20 \
--name vl-crg-nlvr2

---

### 6️⃣ **Evaluating VL-CRG on NLVR²**

PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/nlvr2.py \
--train train --test test \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --load output/vl-crg-nlvr2.pth \
--tqdm

---

### 7️⃣ Running Ablation Studies (Spatial / Positional Encoding)
Baseline (no spatial or positional encoding):
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py \
--train train --valid val \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --epochs 20 \
--name vl-crg-base
With spatial encoding only:
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py \
--train train --valid val \
--use_spatial \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --epochs 20 \
--name vl-crg-spatial
With spatial + positional encoding:
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py \
--train train --valid val \
--use_spatial --use_positional \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --epochs 20 \
--name vl-crg-positional

---

###8️⃣ Pre-training Task Ablation (CRG + MLM/MRC/OPR)
To compare different pre-training objectives (e.g., CRG vs. CRG + MLM + MRC + SRC + OPR):

Run pre-training:

PYTHONPATH=$PYTHONPATH:./src \
python -u src/pretrain/train_crg.py \
--task pretrain \
--pretrain_tasks mlm,mrc,src,opr,crg \
--loadLXMERT snap/pretrained/model \
--batchSize 32 --epochs 30 \
--output output/vl-crg-pretrain-full
You can ablate by removing tasks (e.g., --pretrain_tasks mlm,crg or --pretrain_tasks crg). After pretraining, fine-tune with the command in step 3️⃣ using the newly saved checkpoint:
--load output/vl-crg-pretrain-full.pth
## 📊 **Performance Metrics**
We evaluate **VL-CRG** on **Visual Commonsense Reasoning (VCR)** using the following metrics:

### **Task-Specific Accuracy**
- **Q → A Accuracy**: Predicting the correct answer given an image and a question.
- **QA → R Accuracy**: Selecting the correct rationale given a question and its answer.
- **Q → AR Accuracy**: Jointly predicting both answer and rationale.

### **Robustness Metrics**
- **Consistency**: Ensures logically related questions about the same image receive non-contradictory answers.
- **Sensitivity**: Measures the model’s ability to adjust predictions when input variations (e.g., reference changes) occur.

---

## 🏆 **Results**
### **VL-CRG vs. Baseline (w/o CRG)**
| Model      | Q → A | QA → R | Q → AR | Consistency | Sensitivity |
|------------|------:|------:|------:|------------:|------------:|
| **VL-CRG** |  75.2 |  77.5 |  58.6 |        80.1 |        72.4 |
| *w/o CRG*  |  63.0 (-12.2) |  70.3 (-7.2) |  47.5 (-11.1) |  68.9 (-11.2) |  61.7 (-10.7) |

---

### 🧠 VL-CRG vs. Baseline (Impact of Spatial and Positional Encoding)

| Description           | Cost | Q→A  | QA→R | Q→AR | Consistency | Sensitivity |
|-----------------------|------|------|------|------|-------------|-------------|
| VL-CRG-base           | 1.00 | 60.4 | 63.8 | 37.5 | 61.0        | 54.8        |
| +Spatial Encoding     | 1.05 | 65.1 | 70.9 | 44.0 | 63.7        | 63.6        |
| +Positional Encoding  | 1.20 | 75.2 | 77.5 | 58.6 | 80.1        | 72.4        |

> 📌 *Table 1: Ablation study on the impact of spatial and positional encoding in VL-CRG.*

---

### 🔁 Pre-training Task Comparison on VCR

| Alternative Pre-training Tasks            | Q→A  | QA→R | Q→AR |
|-------------------------------------------|------|------|------|
| CRG                                       | 75.2 | 77.5 | 58.6 |
| MLM + CRG                                 | 76.5 | 78.1 | 58.8 |
| MLM + MRC + CRG                           | 77.1 | 78.1 | 59.0 |
| MLM + MRC + MRFR + CRG                    | 77.1 | 78.1 | 58.9 |
| MLM + MRC + SRC + CRG                     | 77.2 | 78.2 | 59.3 |
| **MLM + MRC + SRC + OPR + CRG**           | **77.3** | **78.4** | **60.0** |

> 📌 *Table 2: Results of VCR val set using different pre-training task combinations on VL-CRG.*

---

### 🌍 Generalization to VQA and NLVR² (Base Setting)

| Models             | VQA (test-dev) | VQA (test-std) | NLVR² (dev) | NLVR² (testP) |
|--------------------|----------------|----------------|-------------|----------------|
| UNITER [2]         | 72.7           | 72.9           | 77.2        | 77.9           |
| Oscar [1]          | 73.2           | 73.4           | 78.1        | 78.4           |
| ERNIE-ViL [4]      | 73.2           | 73.4           | -           | -              |
| **VL-CRG (Ours)**  | 73.8           | 74.2           | 79.0        | 79.7           |
| X²-VLM [14]        | 80.4           | 80.6           | 89.4        | 89.6           |
| BEiT-3 [15]        | **84.19**      | **84.5**       | **92.58**   | **92.8**       |

> 📌 *Table 3: Performance comparison on VQA and NLVR² under the base model setting.*

---

## 📈 **Convergence Analysis**
The graph below shows the validation accuracy trends during training.

![Convergence Analysis](output/convergence_analysis)

- **Solid lines** indicate VL-CRG performance.
- **Dashed lines** show baseline (w/o CRG) performance.
- CRG pre-training improves **convergence speed** and **final accuracy**.

---

## 📜 **Citation**
If you find this work useful, please cite:
```bibtex
@article{vl-crg2025,
  title={VL-CRG: Causal Relation Graph Pre-Training for Vision-Language Tasks},
  author={Xiangrui Sun et al.},
  journal={ArXiv},
  year={2025}
}
```

---

## 🔍 **Acknowledgments**
This repository is built on **LXMERT**. The original LXMERT repo can be found [here](https://github.com/airsplay/lxmert).
