python main.py --task star --version augment --learning-rate 3e-5 --n-epochs 14 --do-train --debug

# ====== BASELINE METHODS ======
#                                  ____ Pre-training Support Models ____
# python main.py --task star --version baseline --learning-rate 3e-5 --n-epochs 14 --do-train --do-save
# python main.py --task flow --version baseline --learning-rate 3e-5 --n-epochs 14 --do-train --do-save
# python main.py --task rostd --version baseline --learning-rate 1e-5 --n-epochs 14 --do-train --do-save

#                                  _____ Evaluation of Baselines _____
# python main.py --task star --version baseline --temperature 1 --do-eval --method maxprob
# python main.py --task flow --version baseline --temperature 1.5 --do-eval --method maxprob
# python main.py --task rostd --version baseline --temperature 1.8 --do-eval --method maxprob

# python main.py --task star --version baseline --temperature 1 --method entropy
# python main.py --task flow --version baseline --temperature 1.5 --method entropy
# python main.py --task rostd --version baseline --temperature 2.8 --method entropy

# (Examples for other methods; Change --task flag to get other target datasets)
# python main.py --task star --version baseline --batch-size 32 --do-eval --method bert_embed
# python main.py --task star --version baseline --do-eval --method rob_embed
# python main.py --task star --version baseline --batch-size 1 --do-eval --method gradient
# python main.py --task star --version baseline --do-eval --method oracle
# python main.py --task star --version baseline --do-eval --method dropout
# python main.py --task star --version baseline --do-eval --method odin

# ========= AUGMENTATION METHODS ========
#                                  _____ Gold Core Detectors _____
# (Variations among extraction technique)
# python main.py --task star --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data PC --num-matches 24 --batch-size 16 --temperature 1.2
# python main.py --task star --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique tfidf --source-data PC --num-matches 24 --batch-size 16 --temperature 1.2
# python main.py --task star --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique paraphrase --source-data PC --num-matches 24 --batch-size 16 --temperature 1.2
# python main.py --task star --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique encoder --source-data PC --num-matches 24 --batch-size 16 --temperature 1.2

# (Variation among source dataset)
# python main.py --task flow --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data QQP --num-matches 24 --batch-size 16 --temperature 1.2
# python main.py --task flow --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data TM --num-matches 24 --batch-size 16 --temperature 1.2
# python main.py --task flow --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data OSQ --num-matches 24 --batch-size 16 --temperature 1.2
# python main.py --task flow --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data MIX --num-matches 24 --batch-size 16 --temperature 1.2

# (Variations among hyper-parameters)
# python main.py --task rostd --version augment --learning-rate 3e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data QQP --num-matches 16 --batch-size 16 --temperature 1.2
# python main.py --task rostd --version augment --learning-rate 1e-5  --n-epochs 14  --do-train --do-save \
#      --technique glove --source-data QQP --num-matches 24 --batch-size 16 --temperature 1.2
# python main.py --task rostd --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data QQP --num-matches 24 --batch-size 32 --temperature 1.2
# python main.py --task rostd --version augment --learning-rate 1e-5  --n-epochs 7  --do-train --do-save \
#      --technique glove --source-data QQP --num-matches 16 --batch-size 16 --temperature 1.2

#                                  _____ Evaluation of Augmented Models _____
# PYTHONPATH=. python main.py --task star --version augment --do-eval --technique paraphrase
# PYTHONPATH=. python main.py --task star --version augment --do-eval --technique encoder
# PYTHONPATH=. python main.py --task star --version augment --do-eval --technique glove
# PYTHONPATH=. python main.py --task star --version augment --do-eval --technique tfidf
# PYTHONPATH=. python main.py --task star --version augment --do-eval --technique random