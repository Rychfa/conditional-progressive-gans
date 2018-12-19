# conditional-progressive-gans
Train the model:
```
python train.py \
  --mode train \
  --output_dir output/facades_train \
  --max_epochs 200 \
  --input_dir dataset/facades/train \
  --which_direction BtoA
  --batch_size 1
```

